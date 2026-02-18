"""
Test the improved account number matching logic
"""
import db_manager

print("=" * 60)
print("Testing IMPROVED Account Number Matching")
print("=" * 60)

# Clear previous test data and add fresh test case
print("\n📋 Setup: Adding test bank...")
db_manager.add_bank('Test Bank', '9339')

banks = db_manager.get_all_banks()
print(f"✅ Test bank added:")
for bank in banks:
    print(f"   - {bank[1]}: {bank[2]}")

print("\n" + "=" * 60)
print("Testing Edge Case: Slip shows '39', DB has '9339'")
print("=" * 60)

# Test the exact scenario user reported
to_account_num = '39'
print(f"\n1️⃣ Testing with extracted number: '{to_account_num}'")
print(f"   Database has: '9339'")

matched_bank = db_manager.find_bank_by_account_number(to_account_num)
if matched_bank:
    print(f"   ✅ SUCCESS! Matched → {matched_bank}")
else:
    print(f"   ❌ FAILED! No match found")

# Additional edge cases
print("\n" + "=" * 60)
print("Additional Edge Cases")
print("=" * 60)

test_cases = [
    ('9339', '9339', 'Exact match'),
    ('339', '9339', 'Last 3 digits'),
    ('39', '9339', 'Last 2 digits'),
    ('9', '9339', 'Last 1 digit'),
    ('93', '9339', 'Middle digits'),
    ('933', '9339', 'First 3 digits'),
    ('99339', '9339', 'Longer input containing stored'),
]

for test_input, stored_in_db, description in test_cases:
    print(f"\n2️⃣ Test: {description}")
    print(f"   Input: '{test_input}' | Stored: '{stored_in_db}'")
    matched = db_manager.find_bank_by_account_number(test_input)
    result = "✅ MATCH" if matched else "❌ NO MATCH"
    print(f"   {result}")

print("\n" + "=" * 60)
print("Test Completed!")
print("=" * 60)
