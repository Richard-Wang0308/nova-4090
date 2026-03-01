import json
import pandas as pd

# Load JSON data from file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract the required columns and process the data
rows = []
for item in data['data']:
    rows.append({
        'block_number': item['block_number'],
        'timestamp': item['timestamp'],
        'uid': item['uid'],
        'registration_cost': int(item['registration_cost']) / (10**9)
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Write to Excel file
df.to_excel('output.xlsx', index=False, sheet_name='Data')

print("Excel file created successfully: output.xlsx")
print("\nPreview of data:")
print(df)
