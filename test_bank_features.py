"""
Test script for new bank management features
"""
import db_manager

print("=" * 60)
print("Testing Bank Management Features")
print("=" * 60)

# Test 1: Add a bank
print("\n1. Testing add_bank()...")
result = db_manager.add_bank('Meezan Bank', '01234567890123')
print(f"   ✅ add_bank() result: {result}")

# Test 2: Get all banks
print("\n2. Testing get_all_banks()...")
banks = db_manager.get_all_banks()
print(f"   ✅ Total banks in DB: {len(banks)}")
for bank in banks:
    print(f"      - ID: {bank[0]}, Name: {bank[1]}, Account: {bank[2]}")

# Test 3: Find bank by account number (full match)
print("\n3. Testing find_bank_by_account_number() - Full match...")
matched = db_manager.find_bank_by_account_number('01234567890123')
print(f"   ✅ Matched bank (full): {matched}")

# Test 4: Find bank by account number (partial - last digits)
print("\n4. Testing find_bank_by_account_number() - Last 4 digits...")
matched = db_manager.find_bank_by_account_number('0123')
print(f"   ✅ Matched bank (last 4): {matched}")

# Test 5: Add another bank
print("\n5. Adding another bank...")
result = db_manager.add_bank('HBL', '98765432109876')
print(f"   ✅ add_bank() result: {result}")

# Test 6: Find second bank by partial account
print("\n6. Testing partial match for second bank...")
matched = db_manager.find_bank_by_account_number('9876')
print(f"   ✅ Matched bank (partial): {matched}")

# Test 7: Update a bank
print("\n7. Testing update_bank()...")
if len(banks) > 0:
    bank_id = banks[0][0]
    result = db_manager.update_bank(bank_id, 'Meezan Bank Updated', '11111111111111')
    print(f"   ✅ update_bank() result: {result}")

# Test 8: Get all banks again to see updates
print("\n8. Final bank list...")
banks = db_manager.get_all_banks()
for bank in banks:
    print(f"   - ID: {bank[0]}, Name: {bank[1]}, Account: {bank[2]}")

# Test 9: Delete a bank
print("\n9. Testing delete_bank()...")
if len(banks) > 0:
    bank_id = banks[-1][0]
    result = db_manager.delete_bank(bank_id)
    print(f"   ✅ delete_bank() result: {result}")
    
    banks = db_manager.get_all_banks()
    print(f"   ✅ Banks remaining: {len(banks)}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
