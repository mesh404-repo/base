"""
Context compaction system for SuperAgent.

Implements intelligent context management like OpenCode/Codex:
1. Token-based overflow detection
2. Tool output pruning (clear old outputs, keep recent)
3. AI-powered conversation compaction (summarization)

This replaces naive sliding window truncation which breaks cache.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from term_sdk import LLM

# =============================================================================
# Constants (matching OpenCode)
# =============================================================================

# Token estimation
APPROX_CHARS_PER_TOKEN = 4

# Context limits
MODEL_CONTEXT_LIMIT = 40_000  # Claude Opus 4.5 context window
OUTPUT_TOKEN_MAX = 16_384  # Max output tokens to reserve
AUTO_COMPACT_THRESHOLD = 0.85  # Trigger compaction at 85% of usable context

# Pruning constants (from OpenCode)
PRUNE_PROTECT = 30_000  # Protect this many tokens of recent tool output
PRUNE_MINIMUM = 5_000  # Only prune if we can recover at least this many tokens
PRUNE_MARKER = "[Old tool result content cleared]"

# Compaction prompts (from Codex)
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Which files were modified and how
- Any errors encountered and how they were resolved

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. Use bullet points and clear sections."""

SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used. Use this to build on the work that has already been done and avoid duplicating work.

Here is the summary from the previous context:

"""


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length (4 chars per token heuristic)."""
    return max(0, len(text or "") // APPROX_CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    tokens = 0
    
    # Content tokens
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                tokens += estimate_tokens(part.get("text", ""))
                # Images count as ~1000 tokens roughly
                if part.get("type") == "image_url":
                    tokens += 1000
    
    # Tool calls tokens (function name + arguments)
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        tokens += estimate_tokens(func.get("name", ""))
        tokens += estimate_tokens(func.get("arguments", ""))
    
    # Role overhead (~4 tokens)
    tokens += 4
    
    return tokens


def estimate_total_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# =============================================================================
# Overflow Detection
# =============================================================================

def get_usable_context() -> int:
    """Get usable context window (total - reserved for output)."""
    return MODEL_CONTEXT_LIMIT


def is_overflow(total_tokens: int, threshold: float = AUTO_COMPACT_THRESHOLD) -> bool:
    """Check if context is overflowing based on token count."""
    usable = get_usable_context()
    return total_tokens > usable * threshold


def needs_compaction(messages: List[Dict[str, Any]]) -> bool:
    """Check if messages need compaction."""
    total_tokens = estimate_total_tokens(messages)
    return is_overflow(total_tokens)


# =============================================================================
# Tool Output Pruning
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [compaction] {msg}", file=sys.stderr, flush=True)


def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    protect_last_turns: int = 2,
) -> List[Dict[str, Any]]:
    """
    Prune old tool outputs to save tokens.
    
    Strategy (exactly like OpenCode compaction.ts lines 49-89):
    1. Go backwards through messages
    2. Skip first 2 user turns (most recent)
    3. Accumulate tool output tokens
    4. Once we've accumulated PRUNE_PROTECT (40K) tokens, start marking for prune
    5. Only actually prune if we can recover > PRUNE_MINIMUM (20K) tokens
    
    Args:
        messages: List of messages
        protect_last_turns: Number of recent user turns to skip (default: 2)
        
    Returns:
        Messages with old tool outputs pruned (content replaced with PRUNE_MARKER)
    """
    if not messages:
        return messages
    
    total = 0  # Total tool output tokens seen (going backwards)
    pruned = 0  # Tokens that will be pruned
    to_prune: List[int] = []  # Indices to prune
    turns = 0  # User turn counter
    
    # Go backwards through messages (like OpenCode)
    for msg_index in range(len(messages) - 1, -1, -1):
        msg = messages[msg_index]
        
        # Count user turns
        if msg.get("role") == "user":
            turns += 1
        
        # Skip the first N user turns (most recent)
        if turns < protect_last_turns:
            continue
        
        # Process tool messages
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            
            # Skip already pruned
            if content == PRUNE_MARKER:
                # Already compacted, stop here (like OpenCode: break loop)
                break
            
            estimate = estimate_tokens(content)
            total += estimate
            
            # Once we've accumulated more than PRUNE_PROTECT tokens,
            # start marking older outputs for pruning
            if total > PRUNE_PROTECT:
                pruned += estimate
                to_prune.append(msg_index)
    
    _log(f"Prune scan: {total} total tokens, {pruned} prunable")
    
    # Only prune if we can recover enough tokens
    if pruned <= PRUNE_MINIMUM:
        _log(f"Prune skipped: only {pruned} tokens recoverable (min: {PRUNE_MINIMUM})")
        return messages
    
    _log(f"Pruning {len(to_prune)} tool outputs, recovering ~{pruned} tokens")
    
    # Create new messages with pruned content
    indices_to_prune = set(to_prune)
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_prune:
            result.append({
                **msg,
                "content": PRUNE_MARKER,
            })
        else:
            result.append(msg)
    
    return result


# =============================================================================
# AI Compaction
# =============================================================================

# First N messages to always keep intact (including system prompt)
PROTECTED_MESSAGE_COUNT = 2


def _find_messages_to_compact(
    messages: List[Dict[str, Any]],
    target_tokens: int,
) -> tuple[int, int]:
    """
    Find the range of messages that need to be compacted.
    
    The first PROTECTED_MESSAGE_COUNT messages (including system prompt) are always kept.
    Calculates how many messages starting from index PROTECTED_MESSAGE_COUNT need to be 
    summarized to bring the total under target_tokens.
    
    Args:
        messages: Current message history
        target_tokens: Target token count to get under
        
    Returns:
        Tuple of (start_index, count) - start index and number of messages to compact
    """
    # Split messages into protected (first 3) and compactable (rest)
    protected_messages = messages[:PROTECTED_MESSAGE_COUNT]
    compactable_messages = messages[PROTECTED_MESSAGE_COUNT:]
    
    # Calculate tokens for each section
    protected_tokens = estimate_total_tokens(protected_messages)
    compactable_tokens = estimate_total_tokens(compactable_messages)
    total_tokens = protected_tokens + compactable_tokens
    
    # Check if we need compaction at all
    if total_tokens <= target_tokens:
        return (0, 0)
    
    # Not enough messages to compact (need at least 2 recent to keep)
    if len(compactable_messages) <= 2:
        return (0, 0)
    
    # Calculate available space for kept messages after compaction
    # Final context = protected_tokens + summary_tokens + kept_tokens
    # We want: protected_tokens + summary_tokens + kept_tokens <= target_tokens
    SUMMARY_TOKEN_ESTIMATE = 2000
    max_kept_tokens = target_tokens - protected_tokens - SUMMARY_TOKEN_ESTIMATE
    
    if max_kept_tokens <= 0:
        # Can't fit even with full compaction, compact everything except last 2
        return (PROTECTED_MESSAGE_COUNT, len(compactable_messages) - 2)
    
    # Calculate how many tokens we need to remove from compactable section
    tokens_to_remove = compactable_tokens - max_kept_tokens
    
    if tokens_to_remove <= 0:
        return (0, 0)
    
    # Accumulate tokens from compactable messages until we have enough to remove
    accumulated = 0
    messages_to_compact = 0
    
    for msg in compactable_messages:
        accumulated += estimate_message_tokens(msg)
        messages_to_compact += 1
        
        if accumulated >= tokens_to_remove:
            break
    
    # Don't compact ALL messages - leave at least the last 2
    max_to_compact = len(compactable_messages) - 2
    messages_to_compact = min(messages_to_compact, max(0, max_to_compact))
    
    return (PROTECTED_MESSAGE_COUNT, messages_to_compact)


def run_compaction(
    llm: "LLM",
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: Optional[str] = None,
    target_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compact conversation history using AI summarization.
    
    Process:
    1. Keep first PROTECTED_MESSAGE_COUNT messages intact (including system prompt)
    2. Calculate how many messages after that need to be summarized to fit under threshold
    3. Summarize only those messages
    4. Keep remaining recent messages intact
    5. Create new message list:
       - First PROTECTED_MESSAGE_COUNT messages (unchanged)
       - Summary as user message (with prefix)
       - Remaining recent messages (unchanged)
    
    Args:
        llm: LLM client for summarization
        messages: Current message history
        system_prompt: Original system prompt to preserve
        model: Model to use (defaults to current)
        target_tokens: Target token count (defaults to 70% of usable context)
        
    Returns:
        Compacted message list with protected and recent messages preserved
    """
    _log("Starting AI compaction...")
    
    # Calculate target tokens if not specified
    if target_tokens is None:
        usable = get_usable_context()
        target_tokens = int(usable * 0.75)
    
    # Find which messages to compact
    compact_start, num_to_compact = _find_messages_to_compact(messages, target_tokens)
    
    if num_to_compact == 0:
        _log("No messages need compaction")
        return messages
    
    # Split messages into three parts:
    # 1. Protected messages (first PROTECTED_MESSAGE_COUNT) - always kept
    # 2. Messages to compact (starting from compact_start)
    # 3. Messages to keep (recent ones after compacted section)
    protected_messages = messages[:PROTECTED_MESSAGE_COUNT]
    messages_to_keep = messages[compact_start + num_to_compact:]
    
    protected_tokens = estimate_total_tokens(protected_messages)
    # compact_tokens = estimate_total_tokens(messages_to_compact)
    keep_tokens = estimate_total_tokens(messages_to_keep)
    
    _log(f"Protected: {len(protected_messages)} messages ({protected_tokens} tokens)")
    # _log(f"Compacting: {num_to_compact} messages ({compact_tokens} tokens)")
    _log(f"Keeping: {len(messages_to_keep)} messages ({keep_tokens} tokens)")
    
    
    compacted = list(protected_messages)  # Copy protected messages    
    
    compacted.extend(messages_to_keep)
    
    final_tokens = estimate_total_tokens(compacted)
    _log(f"Final context: {final_tokens} tokens (target: {target_tokens})")
    
    return compacted

# =============================================================================
# Main Context Management
# =============================================================================

def manage_context(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    llm: "LLM",
    force_compaction: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main context management function.
    
    Called before each LLM request to ensure context fits.
    
    Strategy:
    1. Estimate current token usage
    2. If under threshold, return as-is
    3. Try pruning old tool outputs first
    4. If still over threshold, run AI compaction
    
    Args:
        messages: Current message history
        system_prompt: Original system prompt (preserved through compaction)
        llm: LLM client (for compaction)
        force_compaction: Force compaction even if under threshold
        
    Returns:
        Managed message list (possibly compacted)
    """
    total_tokens = estimate_total_tokens(messages)
    usable = get_usable_context()
    usage_pct = (total_tokens / usable) * 100
    
    _log(f"Context: {total_tokens} tokens ({usage_pct:.1f}%)")
    
    # Check if we need to do anything
    if not force_compaction and not is_overflow(total_tokens):
        return messages
    
    _log(f"Context overflow detected, managing...")
    
    # Step 1: Try pruning old tool outputs
    pruned = prune_old_tool_outputs(messages)
    pruned_tokens = estimate_total_tokens(pruned)
    
    if not is_overflow(pruned_tokens) and not force_compaction:
        _log(f"Pruning sufficient: {total_tokens} -> {pruned_tokens} tokens")
        return pruned
    
    # Step 2: Run AI compaction
    _log(f"Pruning insufficient ({pruned_tokens} tokens), running AI compaction...")
    compacted = run_compaction(llm, pruned, system_prompt)
    compacted_tokens = estimate_total_tokens(compacted)
    
    _log(f"Compaction result: {total_tokens} -> {compacted_tokens} tokens")
    
    return compacted
