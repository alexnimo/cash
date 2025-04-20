import json
import logging
import re

logger = logging.getLogger(__name__)

# Regex for control characters except specific allowed ones (like \n, \r, \t, \b, \f)
# JSON spec allows specific escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
# We target ASCII control chars (0-31) except common whitespace escapes.
# This regex aims to match characters that are *not* printable ASCII (32-126)
# and *not* one of the specifically allowed JSON escapes or basic whitespace.
# We will handle this via direct iteration for more control instead of regex.

# --- Helper Function --- 
def lint_json(json_string: str) -> dict:
    """
    Provides detailed error information for a JSON string.
    Returns a dictionary with 'error', 'message', 'line', 'column', 'position',
    and a 'readable_error' string, or an empty dict if no error.
    """
    try:
        json.loads(json_string)
        return {}  # No error
    except json.JSONDecodeError as e:
        return {
            'error': type(e).__name__,
            'message': e.msg,
            'line': e.lineno,
            'column': e.colno,
            'position': e.pos,
            'readable_error': f"{e.msg} at line {e.lineno} column {e.colno} (char {e.pos})"
        }
    except Exception as e:
        # Catch other potential errors during linting itself
        logger.error(f"Unexpected error during JSON linting: {e}")
        return {
            'error': type(e).__name__,
            'message': str(e),
            'line': None,
            'column': None,
            'position': None,
            'readable_error': f"Unexpected linting error: {str(e)}"
        }

# --- Main Fixing Function ---
def fix_json(json_string: str, max_attempts: int = 10) -> tuple[str | None, list[str]]:
    """
    Attempts to fix common JSON errors iteratively.
    
    Args:
        json_string: The potentially broken JSON string.
        max_attempts: Maximum number of fixing attempts.
        
    Returns:
        Tuple containing the fixed JSON string (or None if unfixable) 
        and a list of fixes applied.
    """
    if not isinstance(json_string, str):
        logger.error(f"fix_json received non-string input type: {type(json_string)}")
        return None, ["Input was not a string"]
        
    fixed_json = json_string
    fixes_applied = []
    attempts = 0
    last_fix_state = ""
    
    # STEP 0: Pre-processing - Remove invalid control characters and normalize escapes
    original_len = len(fixed_json)
    cleaned_chars = []
    i = 0
    while i < len(fixed_json):
        char = fixed_json[i]
        # Keep normal printable ASCII chars
        if 32 <= ord(char) <= 126:
            # Check for escaped characters
            if char == '\\':
                # Check next character for valid escape sequence
                if i + 1 < len(fixed_json):
                    next_char = fixed_json[i+1]
                    # Keep valid JSON escapes
                    if next_char in '"\\/bfnrt': 
                        cleaned_chars.append(char) # Keep backslash
                        cleaned_chars.append(next_char) # Keep escaped char
                        i += 1 # Move past the escaped char
                    # Handle unicode escapes (keep as is for now)
                    elif next_char == 'u' and i + 5 < len(fixed_json): 
                         # Basic check for \uXXXX format
                        if all(c in '0123456789abcdefABCDEF' for c in fixed_json[i+2:i+6]):
                            cleaned_chars.append(char)
                            cleaned_chars.append(next_char)
                            cleaned_chars.extend(list(fixed_json[i+2:i+6]))
                            i += 5
                        else: # Invalid unicode escape, just keep backslash?
                           cleaned_chars.append(char) 
                           # Maybe skip the 'u' and subsequent chars? Risky.
                           # For now, just keep the backslash.
                    else:
                        # Invalid escape sequence, just keep the backslash
                        cleaned_chars.append(char)
                else:
                     # Dangling backslash at the end, keep it
                     cleaned_chars.append(char)
            else:
                # Not a backslash, just a normal printable char
                cleaned_chars.append(char)
        # Keep specific allowed whitespace/control chars
        elif char in '\n\r\t':
            cleaned_chars.append(char)
        # Skip other control characters (ASCII 0-31 excluding \n, \r, \t)
        # We also skip DEL (127)
        
        i += 1 # Move to the next character

    fixed_json = ''.join(cleaned_chars)
    if len(fixed_json) != original_len:
        fix_msg = f"Pre-processing: Removed/normalized {original_len - len(fixed_json)} potentially invalid characters or escape sequences"
        logger.debug(f"DEBUG: {fix_msg}")
        fixes_applied.append(fix_msg)

    
    while attempts < max_attempts:
        current_state = fixed_json
        attempts += 1
        logger.debug(f"DEBUG: Attempt {attempts}: Starting fix cycle. Current length: {len(fixed_json)}")
        
        try:
            # Try parsing the JSON as is
            json.loads(fixed_json)
            logger.info(f"JSON successfully parsed after {attempts} fix attempts")
            return fixed_json, fixes_applied
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in attempt {attempts}: {e}")
            error_info = lint_json(fixed_json)
            error_message = str(e)
            error_pos = e.pos # Position of the error
            
            readable_error = error_info.get('readable_error', f"General error: {error_message}")
            logger.debug(f"DEBUG: Attempt {attempts}: Linted error: '{readable_error}' at position {error_pos}")
            
            # --- STEP 1: Regex-based structural fixes ---
            original_fixed_json = fixed_json
            fix_applied_this_step = False

            # Fix 1: Missing comma between string and opening bracket: "some string"[ -> "some string",[
            # Ensure we don't add a comma if one already exists before the bracket
            fixed_json_new = re.sub(r'("\s*)\[', r'\1", [', fixed_json)
            if fixed_json_new != fixed_json:
                 fixes_applied.append(f"Attempt {attempts}: Added missing comma before '[' after a string.")
                 fixed_json = fixed_json_new
                 fix_applied_this_step = True

            # Fix 2: Missing comma between string and opening brace: "some string"{ -> "some string",{
            fixed_json_new = re.sub(r'("\s*)\{', r'\1", {', fixed_json)
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Added missing comma before '{{' after a string.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True

            # Fix 2.5: Cleanup potential double commas introduced by fixes 1 & 2
            fixed_json_new = re.sub(r',\s*,', ',', fixed_json)
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Cleaned up potential double commas.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True

            # Fix 3: Dangling comma before closing bracket: ...,] -> ...]
            fixed_json_new = re.sub(r',(\s*)\]', r'\1]', fixed_json)
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Removed trailing comma before ']'.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True

            # Fix 4: Dangling comma before closing brace: ...,} -> ...}
            fixed_json_new = re.sub(r',(\s*)\}', r'\1}', fixed_json)
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Removed trailing comma before '}}'.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True
            
            # Fix 5: Add missing comma between consecutive strings: "a" "b" -> "a", "b"
            # Be careful not to add inside keys or already comma-separated lists
            fixed_json_new = re.sub(r'(")(\s+)(")', r'\1,\2\3', fixed_json)
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Added potential missing comma between consecutive strings.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True

            # Fix 6: Add missing comma between string and number: "a" 1 -> "a", 1
            fixed_json_new = re.sub(r'(")(\s+)(\d|\-|true|false|null)', r'\1,\2\3', fixed_json) 
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Added potential missing comma between string and number/bool/null.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True

            # Fix 7: Add missing comma between number and string: 1 "a" -> 1, "a"
            fixed_json_new = re.sub(r'(\d|true|false|null)(\s+)(")', r'\1,\2\3', fixed_json) 
            if fixed_json_new != fixed_json:
                fixes_applied.append(f"Attempt {attempts}: Added potential missing comma between number/bool/null and string.")
                fixed_json = fixed_json_new
                fix_applied_this_step = True
                
            # --- STEP 2: Basic Cleanup ---
            original_len = len(fixed_json)
            fixed_json = fixed_json.strip()
            # Only strip trailing comma if it's truly trailing (not part of structure)
            if fixed_json.endswith(','):
                 fixed_json_new = fixed_json[:-1].rstrip()
                 if fixed_json_new != fixed_json:
                    fixes_applied.append(f"Attempt {attempts}: Removed simple trailing comma.")
                    fixed_json = fixed_json_new
                    fix_applied_this_step = True

            # --- STEP 3: Check for unclosed strings ---
            quote_count = 0
            escaped = False
            for i, char in enumerate(fixed_json):
                if char == '"' and not escaped:
                    quote_count += 1
                # Correct escape check: only the char *before* matters
                if i > 0:
                    escaped = (fixed_json[i-1] == '\\') 
                else:
                    escaped = False
            
            if quote_count % 2 != 0:
                # Odd number of quotes likely means unclosed string at the end
                # Avoid adding if the error isn't related to string termination
                if 'Unterminated string' in error_message or fixed_json.rfind('"') < fixed_json.rfind(':'): # Heuristic: last quote is before last colon
                    fixed_json += '"'
                    fix_msg = f"Attempt {attempts}: Added potential missing closing quote at the end."
                    logger.debug(f"DEBUG: {fix_msg}")
                    fixes_applied.append(fix_msg)
                    fix_applied_this_step = True
            
            # --- STEP 4: Balance brackets/braces ---
            stack = []
            in_string = False
            escaped = False
            open_chars = '[{'
            close_chars = ']}'
            matching = {')': '(', ']': '[', '}': '{'} # Matching pairs
            
            for i, char in enumerate(fixed_json):
                # Correct escape check for current character
                if i > 0:
                    is_escaped = (fixed_json[i-1] == '\\')
                else:
                    is_escaped = False

                if char == '"' and not is_escaped:
                    in_string = not in_string
                    continue # Move to next char
                
                if not in_string: # Only track brackets outside strings
                    if char in open_chars:
                        stack.append(char)
                    elif char in close_chars:
                        if stack and stack[-1] == matching[char]:
                            stack.pop()
                        else:
                            # Found a closing bracket without a matching open one
                            # Or mismatch. This is complex to fix automatically and safely.
                            # We will rely on adding missing ones at the end for now.
                            logger.debug(f"DEBUG: Found potential extra or mismatched closing bracket '{char}' at index {i}")
                            pass # Don't modify here, let the end-balancing handle missing opens
            
            # Add missing closing brackets/braces if stack is not empty
            if stack:
                added_chars = ""
                for open_char in reversed(stack):
                    if open_char == '{': added_chars += '}'
                    elif open_char == '[': added_chars += ']'
                
                if added_chars:
                    fixed_json += added_chars
                    fix_msg = f"Attempt {attempts}: Added missing closing brackets/braces: '{added_chars}'"
                    logger.debug(f"DEBUG: {fix_msg}")
                    fixes_applied.append(fix_msg)
                    fix_applied_this_step = True

            # Check if no changes were made in this attempt to prevent infinite loops
            if fixed_json == current_state:
                logger.warning(f"Attempt {attempts}: No changes made this cycle, JSON likely unfixable with current rules. Error: {readable_error}")
                # Check if it's the same state as the *previous* attempt's end state
                if fixed_json == last_fix_state:
                    logger.error(f"Stuck in loop between attempts {attempts-1} and {attempts}. Aborting fix.")
                    return None, fixes_applied
                last_fix_state = fixed_json # Update last state
                # If it's the final attempt and no change, return None
                if attempts == max_attempts:
                     return None, fixes_applied
                else: # Try one more time if not max attempts
                    continue
            else:
                # Reset last fix state if changes were made
                last_fix_state = ""
                    
    # If loop finishes without successful parsing
    logger.error(f"Failed to fix JSON after {max_attempts} attempts.")
    return None, fixes_applied

def clean_json_response(response_text: str) -> str:
    """Cleans the response text to extract the JSON part."""
    # Find the start of the JSON (either { or [)
    start_brace = response_text.find('{')
    start_bracket = response_text.find('[')

    if start_brace == -1 and start_bracket == -1:
        logger.warning("No JSON object or array found in the response text.")
        return "{}" # Return empty object as fallback

    # Determine the actual start position
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_index = start_brace
        opening_char = '{'
        closing_char = '}'
    else:
        start_index = start_bracket
        opening_char = '['
        closing_char = ']'

    # Extract the text from the start index
    json_part = response_text[start_index:]

    # Attempt to find the matching closing bracket/brace
    level = 0
    end_index = -1
    in_string = False
    escaped = False

    for i, char in enumerate(json_part):
        if char == '"' and not escaped:
            in_string = not in_string
        elif not in_string:
            if char == opening_char:
                level += 1
            elif char == closing_char:
                level -= 1
                if level == 0:
                    end_index = i
                    break # Found the balanced end
        
        # Handle escape characters
        escaped = char == '\\' and not escaped

    if end_index != -1:
        # Found a balanced structure
        cleaned_text = json_part[:end_index + 1]
        logger.debug(f"Extracted JSON block from index {start_index} to {start_index + end_index}")
        return cleaned_text
    else:
        # Could not find a matching closing bracket/brace
        # Return the substring from the start, maybe fix_json can handle it
        logger.warning(f"Could not find matching closing bracket '{closing_char}' for opening '{opening_char}' starting at index {start_index}. Returning potentially truncated JSON.")
        return json_part