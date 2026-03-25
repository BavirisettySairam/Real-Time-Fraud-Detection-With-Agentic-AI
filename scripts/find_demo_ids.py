import pandas as pd

t = pd.read_csv('data/train_transaction.csv')
ident = pd.read_csv('data/train_identity.csv')
df = t.merge(ident, on='TransactionID', how='left')
df['hour'] = (df['TransactionDT'] % 86400) // 3600

print(f'Total rows: {len(df)}, fraud: {df.isFraud.sum()}')

# 1. Routine Purchase: legit, amt 30-100, daytime, ProductCD=W
r1 = df[(df.isFraud==0) & (df.TransactionAmt>=30) & (df.TransactionAmt<=100) & (df.ProductCD=='W') & (df.hour>=9) & (df.hour<=17)]
print(f'\n1. Routine Purchase candidates: {len(r1)}')
if len(r1) > 0:
    row = r1.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}, hour={int(row.hour)}')

# 2. Recurring Subscription: legit, amt 5-20, ProductCD=S
r2 = df[(df.isFraud==0) & (df.TransactionAmt>=5) & (df.TransactionAmt<=20) & (df.ProductCD=='S')]
print(f'\n2. Subscription candidates: {len(r2)}')
if len(r2) > 0:
    row = r2.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}')

# 3. High-Value Midnight: fraud, amt>2000, hour 0-4
r3 = df[(df.isFraud==1) & (df.TransactionAmt>2000) & (df.hour>=0) & (df.hour<=4)]
print(f'\n3. Midnight Fraud candidates: {len(r3)}')
if len(r3) > 0:
    row = r3.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}, hour={int(row.hour)}')

# 4. Gift Card Fraud: fraud, ProductCD=C
r4 = df[(df.isFraud==1) & (df.ProductCD=='C')]
print(f'\n4. Gift Card Fraud candidates: {len(r4)}')
if len(r4) > 0:
    row = r4.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}')

# 5. Borderline: legit, amt 200-500, evening hours
r5 = df[(df.isFraud==0) & (df.TransactionAmt>=200) & (df.TransactionAmt<=500) & (df.hour>=18) & (df.hour<=23)]
print(f'\n5. Borderline candidates: {len(r5)}')
if len(r5) > 0:
    row = r5.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}, hour={int(row.hour)}')

# 6. Agent Disagreement: legit, high-value daytime
r6 = df[(df.isFraud==0) & (df.TransactionAmt>2000) & (df.hour>=9) & (df.hour<=17)]
print(f'\n6. Agent Disagreement candidates: {len(r6)}')
if len(r6) > 0:
    row = r6.iloc[0]
    print(f'   ID={int(row.TransactionID)}, Amt={row.TransactionAmt}, Product={row.ProductCD}, hour={int(row.hour)}')
