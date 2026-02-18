"""
Test the FINAL account number matching logic
Focus: Extract visible digits and match by ending
"""
import db_manager

print("=" * 70)
print("Testing FINAL Account Number Matching Logic")
print("Extract visible digits → Match by ending")
print("=" * 70)

# Setup test banks
print("\n📋 Setup: Adding test banks...")
banks_to_add = [
    ('Meezan Bank', '03010123456789'),
    ('HBL', '00123456789012'),
    ('Allied Bank', '99887766554433'),
    ('Test Bank 1', '9339'),
    ('Test Bank 2', '1234567890232'),
]

for bank_name, account_num in banks_to_add:
    db_manager.add_bank(bank_name, account_num)

banks = db_manager.get_all_banks()
print(f"✅ Added {len(banks_to_add)} test banks:")
for bank in banks:
    print(f"   - {bank[1]}: {bank[2]}")

print("\n" + "=" * 70)
print("Test Cases: Masked Account Numbers")
print("=" * 70)

# Test cases with masked numbers (like real slips)
test_cases = [
    # (VLM_Output, Expected_Bank, Description)
    ('******232', 'Test Bank 2', 'Masked with asterisks - last 3 digits'),
    ('XXXX-XXXX-XX32', 'Test Bank 2', 'Masked with X - last 2 digits'),
    ('***-***-9339', 'Test Bank 1', 'Masked with asterisks and dashes'),
    ('****6789', 'Meezan Bank', 'Partial masking - last 4 digits'),
    ('00123456789012', 'HBL', 'Full account number'),
    ('789012', 'HBL', 'Last 6 digits only'),
    ('4433', 'Allied Bank', 'Last 4 digits'),
    ('***-***-4433', 'Allied Bank', 'Masked with dashes'),
    ('XXXXXXXXXX4433', 'Allied Bank', 'Masked with X characters'),
    ('39', 'Test Bank 1', 'Just last 2 digits'),
    ('232', 'Test Bank 2', 'Just last 3 digits'),
]

print("\n")
results = []
for vlm_output, expected_bank, description in test_cases:
    print(f"🔍 Test: {description}")
    print(f"   VLM Output: '{vlm_output}'")
    print(f"   Expected: {expected_bank}")
    
    matched_bank = db_manager.find_bank_by_account_number(vlm_output)
    
    if matched_bank:
        status = "✅ PASS" if matched_bank == expected_bank else "⚠️ WRONG MATCH"
        print(f"   Result: {matched_bank}")
        print(f"   {status}")
        results.append((description, status == "✅ PASS"))
    else:
        print(f"   Result: No match found")
        print(f"   ❌ FAIL")
        results.append((description, False))
    
    print()

print("=" * 70)
print("Test Summary")
print("=" * 70)

passed = sum(1 for _, success in results if success)
total = len(results)

for desc, success in results:
    status = "✅" if success else "❌"
    print(f"{status} {desc}")

print(f"\nTotal: {passed}/{total} tests passed")
print("=" * 70)

if passed == total:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️ {total - passed} test(s) failed")

print("=" * 70)
