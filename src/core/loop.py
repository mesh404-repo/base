"""
Main agent loop - the heart of the SuperAgent system.

Implements the agentic loop that:
1. Receives instruction from term_sdk context
2. Calls LLM with tools
3. Executes tool calls
4. Loops until task is complete
5. Emits JSONL events throughout

Context management strategy (like OpenCode/Codex):
- Token-based overflow detection (not message count)
- Tool output pruning (clear old outputs first)
- AI compaction when needed (summarize conversation)
- Stable system prompt for cache hits
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from term_sdk import AgentContext, LLMError, CostLimitExceeded

from src.output.jsonl import (
    emit,
    next_item_id,
    reset_item_counter,
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemStartedEvent,
    ItemCompletedEvent,
    ErrorEvent,
    make_agent_message_item,
    make_command_execution_item,
    make_file_change_item,
)
from src.prompts.system import get_system_prompt
from src.utils.truncate import middle_out_truncate, APPROX_BYTES_PER_TOKEN

if TYPE_CHECKING:
    from term_sdk import LLM
    from src.tools.registry import ToolRegistry


# =============================================================================
# Constants (matching OpenCode)
# =============================================================================

# Token estimation
APPROX_CHARS_PER_TOKEN = 4

# Context limits
MODEL_CONTEXT_LIMIT = 4_000  # Claude Opus 4.5 context window
OUTPUT_TOKEN_MAX = 16_384  # Max output tokens to reserve
AUTO_COMPACT_THRESHOLD = 0.85  # Trigger compaction at 85% of usable context

# Pruning constants (from OpenCode)
PRUNE_PROTECT = 40_000  # Protect this many tokens of recent tool output
PRUNE_MINIMUM = 20_000  # Only prune if we can recover at least this many tokens
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

# First N messages to always keep intact (including system prompt)
PROTECTED_MESSAGE_COUNT = 2


# =============================================================================
# Logging
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [loop] {msg}", file=sys.stderr, flush=True)


def _log_compaction(msg: str) -> None:
    """Log compaction messages to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [compaction] {msg}", file=sys.stderr, flush=True)


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
    
    _log_compaction(f"Prune scan: {total} total tokens, {pruned} prunable")
    
    # Only prune if we can recover enough tokens
    if pruned <= PRUNE_MINIMUM:
        _log_compaction(f"Prune skipped: only {pruned} tokens recoverable (min: {PRUNE_MINIMUM})")
        return messages
    
    _log_compaction(f"Pruning {len(to_prune)} tool outputs, recovering ~{pruned} tokens")
    
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
    _log_compaction("Starting AI compaction...")
    
    # Calculate target tokens if not specified
    if target_tokens is None:
        usable = get_usable_context()
        target_tokens = int(usable * 0.75)
    
    # Find which messages to compact
    compact_start, num_to_compact = _find_messages_to_compact(messages, target_tokens)
    
    if num_to_compact == 0:
        _log_compaction("No messages need compaction")
        return messages
    
    # Split messages into three parts:
    # 1. Protected messages (first PROTECTED_MESSAGE_COUNT) - always kept
    # 2. Messages to compact (starting from compact_start)
    # 3. Messages to keep (recent ones after compacted section)
    protected_messages = messages[:PROTECTED_MESSAGE_COUNT]
    messages_to_compact = messages[compact_start:compact_start + num_to_compact]
    messages_to_keep = messages[compact_start + num_to_compact:]
    
    protected_tokens = estimate_total_tokens(protected_messages)
    compact_tokens = estimate_total_tokens(messages_to_compact)
    keep_tokens = estimate_total_tokens(messages_to_keep)
    
    _log_compaction(f"Protected: {len(protected_messages)} messages ({protected_tokens} tokens)")
    _log_compaction(f"Compacting: {num_to_compact} messages ({compact_tokens} tokens)")
    _log_compaction(f"Keeping: {len(messages_to_keep)} messages ({keep_tokens} tokens)")
    
    # Build compaction request with only the messages to compact
    # Strip cache_control and other metadata - only keep role and content
    compaction_messages = []
    for msg in messages_to_compact:
        clean_msg = {
            "role": msg["role"],
            "content": msg.get("content", ""),
        }
        compaction_messages.append(clean_msg)
    
    compaction_messages.append({
        "role": "user",
        "content": COMPACTION_PROMPT,
    })
    
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
           

            response = llm.chat(
                messages_to_compact,
                model="z-ai/glm-4.7",
                max_tokens=4096,  # Summary should be concise
            )
            
            summary = response.text or ""
            
            if not summary:
                _log_compaction("Compaction failed: empty response")
                return messages
            
            summary_tokens = estimate_tokens(summary)
            _log_compaction(f"Compaction complete: {summary_tokens} token summary")
            
            # Build new message list:
            # 1. Protected messages (first PROTECTED_MESSAGE_COUNT, unchanged)
            # 2. Summary of compacted messages
            # 3. Remaining recent messages (preserved exactly)
            compacted = list(protected_messages)  # Copy protected messages
            compacted.append({"role": "user", "content": SUMMARY_PREFIX + summary})
            
            # Add back the messages we kept (recent ones)
            compacted.extend(messages_to_keep)
            
            final_tokens = estimate_total_tokens(compacted)
            _log_compaction(f"Final context: {final_tokens} tokens (target: {target_tokens})")
            
            return compacted
            
        except Exception as e:
            _log_compaction(f"Compaction failed: {e}")
    
    return messages


# =============================================================================
# Main Context Management
# =============================================================================

def manage_context(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    llm: "LLM",
    summarize_llm: "LLM",
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
    
    _log_compaction(f"Context: {total_tokens} tokens ({usage_pct:.1f}%)")
    
    # Check if we need to do anything
    if not force_compaction and not is_overflow(total_tokens):
        return messages
    
    _log_compaction(f"Context overflow detected, managing...")
    
    # Step 1: Try pruning old tool outputs
    pruned = prune_old_tool_outputs(messages)
    pruned_tokens = estimate_total_tokens(pruned)
    
    if not is_overflow(pruned_tokens) and not force_compaction:
        _log_compaction(f"Pruning sufficient: {total_tokens} -> {pruned_tokens} tokens")
        return pruned
    
    # Step 2: Run AI compaction
    _log_compaction(f"Pruning insufficient ({pruned_tokens} tokens), running AI compaction...")
    compacted = run_compaction(summarize_llm, pruned, system_prompt)
    compacted_tokens = estimate_total_tokens(compacted)
    
    _log_compaction(f"Compaction result: {total_tokens} -> {compacted_tokens} tokens")
    
    return compacted


# =============================================================================
# Prompt Caching
# =============================================================================

def _add_cache_control_to_message(
    msg: Dict[str, Any],
    cache_control: Dict[str, str],
) -> Dict[str, Any]:
    """Add cache_control to a message, converting to multipart if needed."""
    content = msg.get("content")
    
    if isinstance(content, list):
        has_cache = any(
            isinstance(p, dict) and "cache_control" in p
            for p in content
        )
        if has_cache:
            return msg
        
        new_content = list(content)
        for i in range(len(new_content) - 1, -1, -1):
            part = new_content[i]
            if isinstance(part, dict) and part.get("type") == "text":
                new_content[i] = {**part, "cache_control": cache_control}
                break
        return {**msg, "content": new_content}
    
    if isinstance(content, str):
        return {
            **msg,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ],
        }
    
    return msg


def _apply_caching(
    messages: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply prompt caching like OpenCode does:
    - Cache first 2 system messages (stable prefix)
    - Cache last 2 non-system messages (extends cache to cover conversation history)
    
    How Anthropic caching works:
    - Cache is based on IDENTICAL PREFIX
    - A cache_control breakpoint tells Anthropic to cache everything BEFORE it
    - By marking the last messages, we cache the entire conversation history
    - Each new request only adds new messages after the cached prefix
    
    Anthropic limits:
    - Maximum 4 cache_control breakpoints
    - Minimum tokens per breakpoint: 1024 (Sonnet), 4096 (Opus 4.5 on Bedrock)
    
    Reference: OpenCode transform.ts applyCaching()
    """
    if not enabled or not messages:
        return messages
    
    cache_control = {"type": "ephemeral"}
    
    # Separate system and non-system message indices
    system_indices = []
    non_system_indices = []
    
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_indices.append(i)
        else:
            non_system_indices.append(i)
    
    # Determine which messages to cache:
    # 1. First 2 system messages (stable system prompt)
    # 2. Last 2 non-system messages (extends cache to conversation history)
    # Total: up to 4 breakpoints (Anthropic limit)
    indices_to_cache = set()
    
    # Add first 2 system messages
    for idx in system_indices[:2]:
        indices_to_cache.add(idx)
    
    # Add last 2 non-system messages
    for idx in non_system_indices[-2:]:
        indices_to_cache.add(idx)
    
    # Build result with cache_control added to selected messages
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_cache:
            result.append(_add_cache_control_to_message(msg, cache_control))
        else:
            result.append(msg)
    
    cached_system = len([i for i in indices_to_cache if i in system_indices])
    cached_final = len([i for i in indices_to_cache if i in non_system_indices])
    
    if indices_to_cache:
        _log(f"Prompt caching: {cached_system} system + {cached_final} final messages marked ({len(indices_to_cache)} breakpoints)")
    
    return result


# =============================================================================
# Main Agent Loop
# =============================================================================

def run_agent_loop(
    llm: "LLM",
    summarize_llm: "LLM",
    tools: "ToolRegistry",
    ctx: AgentContext,
    config: Dict[str, Any],
) -> None:
    """
    Run the main agent loop.
    
    Args:
        llm: LLM client from term_sdk
        tools: Tool registry with available tools
        ctx: Agent context from term_sdk
        config: Configuration dictionary
    """
    # Reset item counter for fresh session
    reset_item_counter()
    
    # Generate session ID
    session_id = f"sess_{int(time.time() * 1000)}"
    
    # 1. Emit thread.started
    emit(ThreadStartedEvent(thread_id=session_id))
    
    # 2. Emit turn.started
    emit(TurnStartedEvent())
    
    # 3. Build initial messages
    cwd = Path(ctx.cwd)
    system_prompt = get_system_prompt(cwd=cwd)
    
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ctx.instruction},
    ]
    
    # 4. Get initial terminal state
    ctx.log("Getting initial state...")
    initial_result = ctx.shell("pwd && ls -la")
    max_output_tokens = config.get("max_output_tokens", 2500)
    initial_state = middle_out_truncate(initial_result.output, max_tokens=max_output_tokens)
    
    messages.append({
        "role": "user",
        "content": f"Current directory and files:\n```\n{initial_state}\n```",
    })
    
    # 5. Initialize tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    pending_completion = False
    last_agent_message = ""
    
    max_iterations = config.get("max_iterations", 200)
    cache_enabled = config.get("cache_enabled", True)
    
    # 6. Main loop
    iteration = 0
    total_cost = 0.0
    while iteration < max_iterations:
        iteration += 1
        ctx.log(f"Iteration {iteration}/{max_iterations}")
        
        try:
            # ================================================================
            # Context Management (replaces sliding window)
            # ================================================================
            # Check token usage and apply pruning/compaction if needed
            context_messages = manage_context(
                messages=messages,
                system_prompt=system_prompt,
                llm=llm,
                summarize_llm=summarize_llm,
            )
            
            # If compaction happened, update our messages reference
            if len(context_messages) < len(messages):
                ctx.log(f"Context compacted: {len(messages)} -> {len(context_messages)} messages")
                messages = context_messages
            
            # ================================================================
            # Apply caching (system prompt only for stability)
            # ================================================================
            cached_messages = _apply_caching(context_messages, enabled=cache_enabled)
            
            # Get tool specs
            tool_specs = tools.get_tools_for_llm()
            
            # ================================================================
            # Call LLM with retry logic
            # ================================================================
            max_retries = 5
            response = None
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    response = llm.chat(
                        cached_messages,
                        model="anthropic/claude-opus-4.5",
                        tools=tool_specs,
                        max_tokens=config.get("max_tokens", 16384),
                        extra_body={
                            "reasoning": {"effort": config.get("reasoning_effort", "xhigh")},
                        },
                    )

                    total_cost += response.cost
                    
                    # Track token usage from response
                    if hasattr(response, "tokens") and response.tokens:
                        tokens = response.tokens
                        if isinstance(tokens, dict):
                            total_input_tokens += tokens.get("input", 0)
                            total_output_tokens += tokens.get("output", 0)
                            total_cached_tokens += tokens.get("cached", 0)
                    
                    break  # Success, exit retry loop
                    
                except CostLimitExceeded:
                    raise  # Don't retry cost limit errors
                    
                except LLMError as e:
                    last_error = e
                    error_msg = str(e.message) if hasattr(e, 'message') else str(e)
                    ctx.log(f"LLM error (attempt {attempt}/{max_retries}): {e.code} - {error_msg}")
                    
                    # Don't retry authentication errors
                    # if e.code in ("authentication_error", "invalid_api_key"):
                    #     raise
                    
                    # Check if it's a retryable error
                    # is_retryable = any(x in error_msg.lower() for x in [
                    #     "504", "timeout", "empty response", "overloaded", "rate_limit"
                    # ])
                    
                    if attempt < max_retries:
                        wait_time = 10 * attempt  # 10s, 20s, 30s, 40s
                        ctx.log(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
                        
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    ctx.log(f"Unexpected error (attempt {attempt}/{max_retries}): {type(e).__name__}: {error_msg}")
                    
                    # is_retryable = any(x in error_msg.lower() for x in ["504", "timeout"])
                    
                    if attempt < max_retries:
                        wait_time = 10 * attempt
                        ctx.log(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
        except CostLimitExceeded as e:
            ctx.log(f"Cost limit exceeded: {e}")
            emit(TurnFailedEvent(error={"message": f"Cost limit exceeded: {e}"}))
            ctx.done()
            return
            
        except LLMError as e:
            ctx.log(f"LLM error (fatal): {e.code} - {e.message}")
            emit(TurnFailedEvent(error={"message": str(e)}))
            ctx.done()
            return
        
        except Exception as e:
            ctx.log(f"Unexpected error (fatal): {type(e).__name__}: {e}")
            emit(TurnFailedEvent(error={"message": str(e)}))
            ctx.done()
            return
        
        # Process response text
        response_text = response.text or ""
        
        if response_text:
            last_agent_message = response_text
            
            # Emit agent message
            item_id = next_item_id()
            emit(ItemCompletedEvent(
                item=make_agent_message_item(item_id, response_text)
            ))
        
        # Check for function calls
        has_function_calls = response.has_function_calls() if hasattr(response, "has_function_calls") else bool(response.function_calls)
        
        if not has_function_calls:
            # No tool calls - agent thinks it's done
            ctx.log("No tool calls in response")
            
            # Always do verification before completing (self-questioning)
            if pending_completion:
                # Agent already verified - complete the task
                ctx.log("Task completion confirmed after self-verification")
                break
            else:
                # First time without tool calls - ask for self-verification
                pending_completion = True
                messages.append({"role": "assistant", "content": response_text})
                
                # Build verification prompt with original instruction
                verification_prompt = f"""<system-reminder>
# Self-Verification Required - CRITICAL

You indicated the task might be complete. Before finishing, you MUST perform a thorough self-verification.

## Original Task (re-read carefully):
{ctx.instruction}

## Self-Verification Checklist:

### 1. Requirements Analysis
- Re-read the ENTIRE original task above word by word
- List EVERY requirement, constraint, and expected outcome mentioned
- Check if there are any implicit requirements you might have missed

### 2. Work Verification  
- For EACH requirement identified, verify it was completed:
  - Run commands to check file contents, test outputs, or verify state
  - Do NOT assume something works - actually verify it
  - If you created code, run it to confirm it works
  - If you modified files, read them back to confirm changes are correct

### 3. Edge Cases & Quality
- Are there any edge cases the task mentioned that you haven't handled?
- Did you follow any specific format/style requirements mentioned?
- Are there any errors, warnings, or issues in your implementation?

### 4. Final Decision
After completing the above verification:
- If EVERYTHING is verified and correct: Summarize what was done and confirm completion
- If ANYTHING is missing or broken: Fix it now using the appropriate tools

## CRITICAL REMINDERS:
- You are running in HEADLESS mode - DO NOT ask questions to the user
- DO NOT ask for confirmation or clarification - make reasonable decisions
- If something is ambiguous, make the most reasonable choice and proceed
- If you find issues during verification, FIX THEM before completing
- Only complete if you have VERIFIED (not assumed) that everything works

Proceed with verification now.
</system-reminder>"""
                
                messages.append({
                    "role": "user", 
                    "content": verification_prompt,
                })
                ctx.log("Requesting self-verification before completion")
                continue
        
        # Reset pending completion flag (agent is still working)
        pending_completion = False
        
        # Add assistant message with tool calls
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
        
        # Build tool_calls for message history
        tool_calls_data = []
        for call in response.function_calls:
            tool_calls_data.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": str(call.arguments) if isinstance(call.arguments, dict) else call.arguments,
                },
            })
        
        if tool_calls_data:
            assistant_msg["tool_calls"] = tool_calls_data
        
        messages.append(assistant_msg)
        
        # Execute each tool call
        for call in response.function_calls:
            tool_name = call.name
            tool_args = call.arguments if isinstance(call.arguments, dict) else {}
            
            _log(f"Executing tool: {tool_name}")
            
            # Emit item.started
            item_id = next_item_id()
            emit(ItemStartedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=f"{tool_name}({tool_args})",
                    status="in_progress",
                )
            ))
            
            # Execute tool
            result = tools.execute(ctx, tool_name, tool_args)
            
            # Truncate output using middle-out (keeps beginning and end)
            output = middle_out_truncate(result.output, max_tokens=max_output_tokens)
            
            # Emit item.completed
            emit(ItemCompletedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=f"{tool_name}",
                    status="completed" if result.success else "failed",
                    aggregated_output=output,
                    exit_code=0 if result.success else 1,
                )
            ))
            
            # Handle image injection
            if result.inject_content:
                # Add image to next user message
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image from {tool_name}:"},
                        result.inject_content,
                    ],
                })
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })

        ctx.log(f"Total cost: ${total_cost}")
        if total_cost >= 4.0:
            break
    # 7. Emit turn.completed
    emit(TurnCompletedEvent(usage={
        "input_tokens": total_input_tokens,
        "cached_input_tokens": total_cached_tokens,
        "output_tokens": total_output_tokens,
    }))
    
    ctx.log(f"Loop complete after {iteration} iterations")
    ctx.log(f"Tokens: {total_input_tokens} input, {total_cached_tokens} cached, {total_output_tokens} output")
    ctx.done()
