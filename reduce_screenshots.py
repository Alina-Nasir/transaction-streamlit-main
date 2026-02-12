"""
Script to identify critical screenshots in USER_MANUAL.md
Focus: SQL Server configuration, connection validation, and troubleshooting
"""

import re

# Count current screenshots in the manual
with open('USER_MANUAL.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all screenshot references
screenshot_pattern = r'!\[.*?\]\((screenshots/.*?\.png)\)'
screenshots = re.findall(screenshot_pattern, content)

print("=" * 80)
print("CRITICAL SCREENSHOTS FOR USER MANUAL")
print("Focus: SQL Server setup, connection validation, and edge case troubleshooting")
print("=" * 80)
print()

# Define CRITICAL screenshots organized by category
categories = {
    "SQL Server Configuration Manager (CRITICAL - Prevents connection failures)": [
        ('screenshots/sql_config_manager.png', 'SQL Server Configuration Manager main window'),
        ('screenshots/tcp_ip_enable.png', 'Right-click TCP/IP → Enable'),
        ('screenshots/tcp_ip_port_config.png', 'IP Addresses tab → IPAll → TCP Port 1433'),
        ('screenshots/sql_service_restart.png', 'SQL Server Services → Right-click → Restart'),
        ('screenshots/sql_service_running.png', 'Service status showing Running with green arrow'),
    ],
    
    "Installer - Connection Validation (CRITICAL - Ensures proper setup)": [
        ('screenshots/installer_welcome.png', 'Installer welcome screen'),
        ('screenshots/installer_auth_selection.png', 'Windows Auth vs SQL Server Auth selection'),
        ('screenshots/installer_sql_config.png', 'Host, port, database name configuration'),
        ('screenshots/folder_selection.png', 'Monitored folder selection for batch processing'),
        ('screenshots/install_complete.png', 'Installation successful screen'),
    ],
    
    "Minimal UI Reference (For basic orientation only)": [
        ('screenshots/main_interface.png', 'Main dashboard after successful launch'),
        ('screenshots/process_transactions.png', 'Process Transactions page overview'),
        ('screenshots/view_database.png', 'View Database page overview'),
    ],
}

# Print organized list
total_screenshots = 0
for category, screenshot_list in categories.items():
    print(f"\n{category}")
    print("-" * 80)
    for path, description in screenshot_list:
        total_screenshots += 1
        status = "✓ ALREADY IN MANUAL" if path in screenshots else "✗ NEEDS TO BE ADDED"
        print(f"  {total_screenshots}. {path}")
        print(f"     Description: {description}")
        print(f"     Status: {status}")
    print()

print("=" * 80)
print(f"TOTAL CRITICAL SCREENSHOTS: {total_screenshots}")
print("=" * 80)
print()

# Analysis of current manual
print("CURRENT MANUAL ANALYSIS:")
print("-" * 80)
print(f"Screenshots currently referenced: {len(set(screenshots))}")
print()

# List screenshots in manual
print("Screenshots found in USER_MANUAL.md:")
for i, path in enumerate(sorted(set(screenshots)), 1):
    print(f"  {i}. {path}")

print()
print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
The manual should focus on:
1. SQL Server Configuration Manager steps (5 screenshots) - MOST CRITICAL
2. Installer connection validation (7 screenshots) - Ensures proper setup
3. Prerequisites and troubleshooting (3 screenshots) - Prevents failures
4. Minimal UI reference (3 screenshots) - Basic orientation

Total: 18 critical screenshots

These screenshots document edge cases that cause system failures:
- TCP/IP not enabled → Cannot connect to database
- Wrong port configuration → Connection timeout
- Service not running → Application won't start
- Wrong credentials → Authentication failure
- Database initialization errors → App crashes on first launch

Non-essential screenshots (basic UI like file upload, search filters, etc.) 
should be removed as they don't prevent system failures.
""")

