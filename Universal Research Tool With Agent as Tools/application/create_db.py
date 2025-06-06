import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("customer_data.db")
cursor = conn.cursor()

# Drop old tables
cursor.executescript("""
DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS user_interactions;
DROP TABLE IF EXISTS user_transactions;
""")

# Create new tables
cursor.executescript("""
CREATE TABLE user (
    UserID TEXT PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    AccountBalance REAL,
    AccountTypes TEXT
);

CREATE TABLE user_interactions (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID TEXT,
    Channel TEXT,
    Interaction TEXT,
    FOREIGN KEY (UserID) REFERENCES user(UserID)
);

CREATE TABLE user_transactions (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID TEXT,
    Type TEXT,
    Amount REAL,
    Date TEXT,
    Counterparty TEXT,
    AccountNumber TEXT,
    Description TEXT,
    FOREIGN KEY (UserID) REFERENCES user(UserID)
);
""")

# Insert Users
cursor.execute("""INSERT INTO user VALUES ('U001', 'Paul Smith', '922 Michelle Junctions Apt. 072, Murrayton, NY', 41587.72, 'Savings, Credit Card')""")
cursor.execute("""INSERT INTO user VALUES ('U002', 'Charles Melton', '7670 Hernandez Mountain Suite 612, Davidstad, TX', 592492.00, 'Savings, Checking, CD1, CD2, Credit Card')""")
cursor.execute("""INSERT INTO user VALUES ('U003', 'Tiffany Romero', '04906 Kim Ferry Suite 801, Ianburgh, HI', 38702.86, 'Mortgage, Checking, Savings')""")

# Insert Interactions
# === User U001 Interactions (7 rows) ===
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Mobile App', 'Searched for travel credit cards')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Mobile App', 'Viewed Chase Sapphire card benefits')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Mobile App', 'Compared credit card rewards')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Agent', 'Spoke with agent about credit card options')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Agent', 'Discussed travel benefits of credit cards')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Mobile App', 'Checked credit card application status')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U001', 'Mobile App', 'Viewed credit card terms and conditions')""")

# === User U002 Interactions (10 rows) ===
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Branch', 'Visited branch for mortgage consultation')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Mobile App', 'Viewed commercial property listings')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Agent', 'Spoke with mortgage specialist')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Branch', 'Submitted mortgage application documents')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Mobile App', 'Checked mortgage application status')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Agent', 'Discussed commercial property financing options')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Branch', 'Signed mortgage documents')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Mobile App', 'Viewed property investment opportunities')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Agent', 'Finalized mortgage terms')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U002', 'Mobile App', 'Set up mortgage payment schedule')""")

# === User U003 Interactions (8 rows) ===
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Web', 'Searched for high yield savings')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Mobile App', 'Logged into Chase app')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Web', 'Viewed APR details')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Mobile App', 'Set up mobile deposit')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Web', 'Updated security settings')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Mobile App', 'Checked recent transactions')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Web', 'Viewed account statements')""")
cursor.execute("""INSERT INTO user_interactions (UserID, Channel, Interaction) VALUES ('U003', 'Mobile App', 'Set up travel notification')""")

# Insert transactions (no loops, each query separately written)

# === Transactions for User U001 (35 rows for 6 months) ===
# Month 1 - December 2024
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2024-12-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2024-12-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 85.00, '2024-12-10', 'Amazon', '1111222233335555', 'Online shopping')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2024-12-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 45.00, '2024-12-20', 'Spotify', '1111222233335555', 'Music subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 500.00, '2024-12-25', 'Freelance', '1111222233334444', 'Part-time work')""")

# Month 2 - January 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2025-01-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2025-01-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2025-01-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 95.00, '2025-01-18', 'Walmart', '1111222233335555', 'Groceries')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 200.00, '2025-01-20', 'Delta Airlines', '1111222233335555', 'Flight booking')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 300.00, '2025-01-25', 'Freelance', '1111222233334444', 'Project completion')""")

# Month 3 - February 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2025-02-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2025-02-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2025-02-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 75.00, '2025-02-22', 'Target', '1111222233335555', 'Household items')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 150.00, '2025-02-25', 'Marriott', '1111222233335555', 'Hotel booking')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 400.00, '2025-02-28', 'Freelance', '1111222233334444', 'Extra project')""")

# Month 4 - March 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2025-03-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2025-03-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2025-03-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 110.00, '2025-03-25', 'Best Buy', '1111222233335555', 'Electronics')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 180.00, '2025-03-28', 'United Airlines', '1111222233335555', 'Flight booking')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 350.00, '2025-03-30', 'Freelance', '1111222233334444', 'Consulting work')""")

# Month 5 - April 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2025-04-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2025-04-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2025-04-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 150.00, '2025-04-20', 'Amazon', '1111222233335555', 'Holiday shopping')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 200.00, '2025-04-25', 'Hilton', '1111222233335555', 'Hotel booking')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 450.00, '2025-04-28', 'Freelance', '1111222233334444', 'Project completion')""")

# Month 6 - May 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 2000.00, '2025-05-05', 'Payroll', '1111222233334444', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 120.00, '2025-05-06', 'Netflix', '1111222233335555', 'Streaming subscription')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 1500.00, '2025-05-15', 'NYU', '1111222233334444', 'Monthly tuition payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 90.00, '2025-05-25', 'Walmart', '1111222233335555', 'Monthly groceries')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'DR', 250.00, '2025-05-28', 'American Airlines', '1111222233335555', 'Flight booking')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U001', 'CR', 500.00, '2025-05-30', 'Freelance', '1111222233334444', 'Year-end project')""")

# === Transactions for User U002 (33 rows for 6 months) ===
# Month 1 - December 2024
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2024-12-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 8500.00, '2024-12-11', 'Emirates', '2222333344446666', 'Business class flight tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 2500.00, '2024-12-15', 'Four Seasons', '2222333344446666', 'Hotel stay Dubai')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 10000.00, '2024-12-20', 'Commercial Property', '2222333344447777', 'Property maintenance')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 5000.00, '2024-12-25', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# Month 2 - January 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2025-01-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 15000.00, '2025-01-11', 'Commercial Property', '2222333344447777', 'Property down payment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 5000.00, '2025-01-20', 'Investment Fund', '2222333344448888', 'CD investment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 7500.00, '2025-01-25', 'Singapore Airlines', '2222333344446666', 'Business class tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 3000.00, '2025-01-28', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# Month 3 - February 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2025-02-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 7500.00, '2025-02-11', 'Singapore Airlines', '2222333344446666', 'Business class tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 3000.00, '2025-02-18', 'Marina Bay Sands', '2222333344446666', 'Hotel stay Singapore')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 12000.00, '2025-02-20', 'Commercial Property', '2222333344447777', 'Property renovation')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 4000.00, '2025-02-25', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# Month 4 - March 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2025-03-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 5000.00, '2025-03-11', 'Investment Fund', '2222333344448888', 'CD investment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 10000.00, '2025-03-22', 'Commercial Property', '2222333344447777', 'Property renovation')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 9000.00, '2025-03-25', 'Qatar Airways', '2222333344446666', 'Business class tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 3500.00, '2025-03-28', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# Month 5 - April 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2025-04-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 9000.00, '2025-04-11', 'Qatar Airways', '2222333344446666', 'Business class tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 4000.00, '2025-04-20', 'Burj Al Arab', '2222333344446666', 'Hotel stay Dubai')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 15000.00, '2025-04-25', 'Commercial Property', '2222333344447777', 'Property maintenance')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 4000.00, '2025-04-28', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# Month 6 - May 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 25000.00, '2025-05-10', 'Business Income', '2222333344445555', 'Monthly business revenue')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 5000.00, '2025-05-11', 'Investment Fund', '2222333344448888', 'CD investment')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 15000.00, '2025-05-25', 'Commercial Property', '2222333344447777', 'Property maintenance')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'DR', 12000.00, '2025-05-28', 'Emirates', '2222333344446666', 'Business class tickets')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U002', 'CR', 4500.00, '2025-05-30', 'Investment Return', '2222333344448888', 'CD interest payment')""")

# === Transactions for User U003 (35 rows for 6 months) ===
# Month 1 - December 2024
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2024-12-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 300.00, '2024-12-02', 'Amazon', '3333444455557777', 'Online purchase')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 250.00, '2024-12-15', 'Freelance', '3333444455556666', 'Consulting work')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 95.00, '2024-12-20', 'Walmart', '3333444455557777', 'Groceries')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2024-12-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Month 2 - January 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2025-01-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 85.00, '2025-01-05', 'Target', '3333444455557777', 'Shopping')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 300.00, '2025-01-15', 'Freelance', '3333444455556666', 'Project completion')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2025-01-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Month 3 - February 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2025-02-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 120.00, '2025-02-03', 'Best Buy', '3333444455557777', 'Electronics')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 280.00, '2025-02-15', 'Freelance', '3333444455556666', 'Consulting work')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2025-02-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Month 4 - March 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2025-03-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 150.00, '2025-03-05', 'Amazon', '3333444455557777', 'Online shopping')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 320.00, '2025-03-15', 'Freelance', '3333444455556666', 'Extra project')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2025-03-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Month 5 - April 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2025-04-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 200.00, '2025-04-10', 'Amazon', '3333444455557777', 'Holiday shopping')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 350.00, '2025-04-15', 'Freelance', '3333444455556666', 'Year-end project')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2025-04-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Month 6 - May 2025
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 2100.00, '2025-05-01', 'Payroll', '3333444455556666', 'Monthly salary')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 95.00, '2025-05-05', 'Walmart', '3333444455557777', 'Groceries')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'CR', 300.00, '2025-05-15', 'Freelance', '3333444455556666', 'New year project')""")
cursor.execute("""INSERT INTO user_transactions (UserID, Type, Amount, Date, Counterparty, AccountNumber, Description) 
VALUES ('U003', 'DR', 45.00, '2025-05-25', 'Netflix', '3333444455557777', 'Streaming subscription')""")

# Finalize DB
conn.commit()
conn.close()
print("Database 'chase_data.db' created successfully with full dataset.")
