# Testing Guide: Auto-Refresh with Streamlit Fragments

## Quick Testing Checklist

Before rebuilding the installer, verify these 5 critical tests pass:

### ✅ Test 1: Auto-Refresh Works Without User Action
**What to test**: Data updates automatically every 2 seconds  
**Steps**:
1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Navigate to "View Database" page
3. Watch the "Total Records" metric in Database Summary
4. **Expected**: Timestamp or count should update every 2 seconds
5. **Verify**: No need to click refresh button

**Pass Criteria**: Data updates visible without any user interaction

---

### ✅ Test 2: Search Filter Preserved During Auto-Refresh
**What to test**: Search input doesn't clear during auto-refresh  
**Steps**:
1. Go to "View Database" page
2. Type "HBL" (or any bank name) in the search box
3. Verify table shows only filtered results
4. Wait 6 seconds (3 auto-refresh cycles)
5. **Expected**: "HBL" text still in search box
6. **Expected**: Table still shows only HBL transactions

**Pass Criteria**: Search term never disappears, filtered view persists

---

### ✅ Test 3: Date Filters Preserved During Auto-Refresh
**What to test**: Date range selections don't reset during auto-refresh  
**Steps**:
1. Go to "View Database" page
2. Select "From Date" (e.g., 01/01/2024)
3. Select "To Date" (e.g., 31/12/2024)
4. Wait 6 seconds (3 auto-refresh cycles)
5. **Expected**: Both date selections still set
6. **Expected**: Table shows only transactions in date range

**Pass Criteria**: Date filters never reset, filtered data persists

---

### ✅ Test 4: New Transactions Appear Automatically
**What to test**: Batch processor adds transaction → appears within 2 seconds  
**Steps**:
1. Start batch processor service: `python run_batch_service.py`
2. Open Streamlit app and go to "View Database"
3. Note current "Total Records" count
4. Drop a new bank slip image into the batch watch folder
5. Check batch processor logs to confirm processing complete
6. Watch "View Database" page (DON'T click refresh)
7. **Expected**: Within 2 seconds, record count increases by 1
8. **Expected**: New transaction appears at top of table

**Pass Criteria**: New record appears automatically without manual refresh

---

### ✅ Test 5: Export Always Shows Current Data
**What to test**: Export buttons download fresh data after auto-refresh  
**Steps**:
1. Go to "View Database" page with 5 records
2. Click "Download Filtered as CSV"
3. Verify CSV has 5 records
4. Add new transaction via batch processor
5. Wait 2 seconds for auto-refresh
6. Verify table shows 6 records
7. Click "Download Filtered as CSV" again
8. Verify CSV now has 6 records (not 5)

**Pass Criteria**: Export always includes latest data, not stale cache

---

## Debugging Failed Tests

### If Test 1 Fails (No Auto-Refresh)
**Problem**: Fragment not auto-refreshing  
**Check**:
```bash
pip show streamlit  # Must be >= 1.30.0
```
**Fix**: `pip install streamlit>=1.30`

---

### If Test 2 or 3 Fails (Filters Reset)
**Problem**: Filter widgets inside fragment (shouldn't be)  
**Check**: Verify in `streamlit_app.py` line 691-701:
```python
# These MUST be OUTSIDE the @st.fragment decorator
search_term = st.text_input(...)  # Line 697
date_from = st.date_input(...)     # Line 700
date_to = st.date_input(...)       # Line 703
```
**Fix**: Ensure filter widgets are before the `@st.fragment(run_every="2s")` decorator

---

### If Test 4 Fails (New Records Don't Appear)
**Problem**: Database not updating OR fragment not querying fresh data  
**Check**:
1. Batch processor logs: `logs/batch_processor_YYYYMMDD.log`
   - Verify: "Database updated successfully for ..."
2. Database file: Check timestamp of `C:\Users\...\AppData\Roaming\PakistanBankParser\transactions.db`
3. Fragment code: Verify line 717 has `cursor.execute('SELECT * FROM transactions ORDER BY CreatedAt DESC')`

**Fix**: 
- If batch processor not writing: Check batch_processor_service.py `_write_update_notification()`
- If fragment not querying: Ensure no database connection caching

---

### If Test 5 Fails (Export Shows Stale Data)
**Problem**: Export buttons outside fragment (shouldn't be)  
**Check**: Verify in `streamlit_app.py` line 770-817:
```python
# Export buttons MUST be INSIDE the fragment
with export_col1:
    csv_data = filtered_df.to_csv(...)
    st.download_button(...)
```
**Fix**: Ensure all export buttons are inside the `display_data_fragment()` function

---

## Performance Testing

### Monitor Fragment Execution Frequency
Add temporary logging to verify 2-second interval:

```python
@st.fragment(run_every="2s")
def display_data_fragment():
    logger.info(f"Fragment refresh: {datetime.now().strftime('%H:%M:%S.%f')}")
    # ... rest of code
```

**Check logs**: Should see log entries exactly 2 seconds apart

---

### Test with Large Dataset
If you have > 1,000 records, test performance:

1. Add 1,000 test records to database
2. Navigate to "View Database"
3. Monitor CPU usage during auto-refresh
4. **Expected**: CPU spike every 2 seconds (for query + render)
5. **Acceptable**: < 10% CPU usage on average

If performance is poor, increase interval to 5 seconds:
```python
@st.fragment(run_every="5s")  # Instead of "2s"
```

---

## Final Pre-Deploy Checklist

Before running `pyinstaller` and `iscc`, confirm:

- [ ] All 5 tests pass in development environment
- [ ] No thread-safety warnings in logs
- [ ] No "st.rerun() called from background thread" errors
- [ ] Filter inputs never reset during auto-refresh
- [ ] New records appear within 2 seconds
- [ ] Export buttons always show current data
- [ ] Streamlit version >= 1.30.0 in requirements.txt

---

## Build and Deploy

Once all tests pass:

```bash
# 1. Clean previous builds
rm -rf build dist

# 2. Build executable
pyinstaller --clean PakistanBankParser.spec

# 3. Test the built executable (repeat all 5 tests above)
./dist/PakistanBankParser/PakistanBankParser.exe

# 4. If executable tests pass, create installer
iscc setup.iss

# 5. Install on test machine and repeat tests
./installer_output/PakistanBankParser_Setup.exe
```

---

## Success Criteria

✅ **Development**: All 5 tests pass in `streamlit run` mode  
✅ **Executable**: All 5 tests pass in PyInstaller build  
✅ **Installer**: All 5 tests pass after installation on clean machine  
✅ **Production**: Batch processor + auto-refresh work together seamlessly

If any test fails at any stage, **DO NOT proceed** to next stage. Fix the issue first.

---

## Support

If tests fail and you can't identify the issue:

1. Check logs: `logs/batch_processor_YYYYMMDD.log` and `logs/app_YYYYMMDD.log`
2. Verify Streamlit version: `pip show streamlit`
3. Review code changes: `git diff HEAD~1 streamlit_app.py`
4. Compare against reference: `STREAMLIT_FRAGMENTS_AUTO_REFRESH.md`

---

## Summary

This solution uses **Streamlit Fragments** (`@st.fragment(run_every="2s")`), which is:
- ✅ The official Streamlit pattern for auto-refresh
- ✅ Thread-safe (no daemon threads)
- ✅ Preserves user inputs (filters, search)
- ✅ Production-ready (works in PyInstaller + Inno Setup)

Testing these 5 scenarios confirms the implementation is correct before time-consuming installer rebuild.
