"""
Test trailing digit extraction logic
"""
import re

def extract_trailing_digits(account_number_partial):
    """Extract the last continuous group of digits"""
    input_str = str(account_number_partial)
    digit_groups = re.findall(r'\d+', input_str)
    
    if not digit_groups:
        return None
    
    return digit_groups[-1]

# Test cases
test_cases = [
    "01*********39",
    "******232",
    "XXXX-XXXX-XX32",
    "***-***-9339",
    "****6789",
    "00123456789012",
    "***-***-6017",
    "01XXXXXXXX39",
    "9339",
    "39"
]

print("Testing Trailing Digit Extraction:")
print("=" * 60)

for test in test_cases:
    result = extract_trailing_digits(test)
    print(f"Input: {test:20s} → Trailing digits: {result}")

print("\n" + "=" * 60)
print("Expected matches with banks:")
print("  - Habib Metro: 6017")
print("  - Meezan Bank: 9339")
print("=" * 60)

# Simulate matching
banks = [("Habib Metro", "6017"), ("Meezan Bank", "9339")]

for test in test_cases:
    trailing = extract_trailing_digits(test)
    matched = False
    
    for bank_name, account in banks:
        if account.endswith(trailing):
            print(f"✅ {test:20s} → Matched: {bank_name}")
            matched = True
            break
    
    if not matched:
        print(f"❌ {test:20s} → No match")
