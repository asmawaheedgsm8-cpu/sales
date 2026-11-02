import pandas as pd
data = {
    "Name": ["Asma khan", "ALI Raza", "Asma Khan", "Sara Ahmad", "Ali Raza"],
    "Email": ["ASMA@GMAIL.COM", "ALI@GMAIL.COM", "SARA@GMAIL.COM", "asma@gmail.com", "ali@gmail.com"]
}
df = pd.DataFrame(data)

print("BEFORE - Ganda Data:")
print(df)

df['Name'] = df['Name'].str.strip().str.title()
df['Email'] = df['Email'].str.strip().str.lower()
df = df.drop_duplicates()

print("\nAFTER - Saaf Data")
print(df)
df.loc[len(df)] = ['Ayeshsa khan', "ayesha@gmail.com"]
print(df)