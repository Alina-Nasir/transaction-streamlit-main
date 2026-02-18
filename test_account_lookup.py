"""
Test script for account number lookup logic
Simulates the transaction processing flow
"""
import db_manager

print("=" * 60)
print("Testing Account Number Lookup Logic")
print("=" * 60)

# Setup: Add test banks
print("\n📋 Setup: Adding test banks...")
db_manager.add_bank('Meezan Bank', '03010123456789')
db_manager.add_bank('HBL', '00123456789012')
db_manager.add_bank('Allied Bank', '99887766554433')

banks = db_manager.get_all_banks()
print(f"✅ Added {len(banks)} test banks:")
for bank in banks:
    print(f"   - {bank[1]}: {bank[2]}")

print("\n" + "=" * 60)
print("Simulating Transaction Processing")
print("=" * 60)

# Test Case 1: Full account number from slip
print("\n1️⃣ Test: Full account number extracted")
mock_vlm_output_1 = {
    'bankName': 'Meezan Bank',
    'Date': '15/02/2026',
    'TransactionID': 'TXN123456',
    'Amount': '50000 PKR',
    'FromAccount': 'Ali Ahmed',
    'FromAccountNumber': '01234567890',
    'ToAccount': 'Hassan Khan',
    'ToAccountNumber': '03010123456789',  # Full account number
    'PaymentMode': 'Online'
}

to_account_num = mock_vlm_output_1.get('ToAccountNumber', 'Not Found')
print(f"   Extracted ToAccountNumber: {to_account_num}")

if to_account_num and to_account_num != 'Not Found':
    matched_bank = db_manager.find_bank_by_account_number(to_account_num)
    if matched_bank:
        mock_vlm_output_1['ToBankName'] = matched_bank
        print(f"   ✅ Matched → {matched_bank}")
    else:
        mock_vlm_output_1['ToBankName'] = 'Not Found'
        print(f"   ❌ No match found")
else:
    mock_vlm_output_1['ToBankName'] = 'Not Found'
    print(f"   ⚠️ No account number extracted")

# Test Case 2: Partial account number (last digits)
print("\n2️⃣ Test: Partial account number (last 6 digits)")
mock_vlm_output_2 = {
    'bankName': 'HBL',
    'Date': '16/02/2026',
    'TransactionID': 'TXN789012',
    'Amount': '25000 PKR',
    'FromAccount': 'Sara Ali',
    'FromAccountNumber': '99887766554',
    'ToAccount': 'Ahmed Raza',
    'ToAccountNumber': '789012',  # Only last 6 digits
    'PaymentMode': 'Cash'
}

to_account_num = mock_vlm_output_2.get('ToAccountNumber', 'Not Found')
print(f"   Extracted ToAccountNumber: {to_account_num}")

if to_account_num and to_account_num != 'Not Found':
    matched_bank = db_manager.find_bank_by_account_number(to_account_num)
    if matched_bank:
        mock_vlm_output_2['ToBankName'] = matched_bank
        print(f"   ✅ Matched → {matched_bank}")
    else:
        mock_vlm_output_2['ToBankName'] = 'Not Found'
        print(f"   ❌ No match found")
else:
    mock_vlm_output_2['ToBankName'] = 'Not Found'

# Test Case 3: Account number with spaces/dashes
print("\n3️⃣ Test: Account number with spaces/dashes")
mock_vlm_output_3 = {
    'bankName': 'Allied Bank',
    'Date': '16/02/2026',
    'TransactionID': 'TXN345678',
    'Amount': '75000 PKR',
    'FromAccount': 'Bilal Khan',
    'FromAccountNumber': '12345678901',
    'ToAccount': 'Fatima Malik',
    'ToAccountNumber': '9988-7766-5544-33',  # With dashes
    'PaymentMode': 'Online'
}

to_account_num = mock_vlm_output_3.get('ToAccountNumber', 'Not Found')
print(f"   Extracted ToAccountNumber: {to_account_num}")

if to_account_num and to_account_num != 'Not Found':
    matched_bank = db_manager.find_bank_by_account_number(to_account_num)
    if matched_bank:
        mock_vlm_output_3['ToBankName'] = matched_bank
        print(f"   ✅ Matched → {matched_bank}")
    else:
        mock_vlm_output_3['ToBankName'] = 'Not Found'
        print(f"   ❌ No match found")
else:
    mock_vlm_output_3['ToBankName'] = 'Not Found'

# Test Case 4: Unknown account number
print("\n4️⃣ Test: Unknown account number (not in database)")
mock_vlm_output_4 = {
    'bankName': 'Some Bank',
    'Date': '16/02/2026',
    'TransactionID': 'TXN999999',
    'Amount': '10000 PKR',
    'FromAccount': 'Unknown Person',
    'FromAccountNumber': '11111111111',
    'ToAccount': 'Another Person',
    'ToAccountNumber': '55555555555555',  # Not in database
    'PaymentMode': 'Cheque'
}

to_account_num = mock_vlm_output_4.get('ToAccountNumber', 'Not Found')
print(f"   Extracted ToAccountNumber: {to_account_num}")

if to_account_num and to_account_num != 'Not Found':
    matched_bank = db_manager.find_bank_by_account_number(to_account_num)
    if matched_bank:
        mock_vlm_output_4['ToBankName'] = matched_bank
        print(f"   ✅ Matched → {matched_bank}")
    else:
        mock_vlm_output_4['ToBankName'] = 'Not Found'
        print(f"   ⚠️ No match found - User should add this bank in 'Manage Banks'")
else:
    mock_vlm_output_4['ToBankName'] = 'Not Found'

# Test Case 5: No account number extracted
print("\n5️⃣ Test: No account number extracted (Not Found)")
mock_vlm_output_5 = {
    'bankName': 'Test Bank',
    'Date': '16/02/2026',
    'TransactionID': 'TXN111111',
    'Amount': '5000 PKR',
    'FromAccount': 'Test User',
    'FromAccountNumber': '12345678901',
    'ToAccount': 'Test Receiver',
    'ToAccountNumber': 'Not Found',  # VLM couldn't extract it
    'PaymentMode': 'Cash'
}

to_account_num = mock_vlm_output_5.get('ToAccountNumber', 'Not Found')
print(f"   Extracted ToAccountNumber: {to_account_num}")

if to_account_num and to_account_num != 'Not Found':
    matched_bank = db_manager.find_bank_by_account_number(to_account_num)
    if matched_bank:
        mock_vlm_output_5['ToBankName'] = matched_bank
        print(f"   ✅ Matched → {matched_bank}")
    else:
        mock_vlm_output_5['ToBankName'] = 'Not Found'
        print(f"   ❌ No match found")
else:
    mock_vlm_output_5['ToBankName'] = 'Not Found'
    print(f"   ⚠️ No account number to lookup")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"✅ Test 1 (Full): {mock_vlm_output_1['ToBankName']}")
print(f"✅ Test 2 (Partial): {mock_vlm_output_2['ToBankName']}")
print(f"✅ Test 3 (With dashes): {mock_vlm_output_3['ToBankName']}")
print(f"⚠️ Test 4 (Unknown): {mock_vlm_output_4['ToBankName']}")
print(f"⚠️ Test 5 (No extraction): {mock_vlm_output_5['ToBankName']}")

print("\n" + "=" * 60)
print("Account Lookup Logic Test Completed!")
print("=" * 60)
