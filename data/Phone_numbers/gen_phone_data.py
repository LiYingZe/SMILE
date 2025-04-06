import csv
from faker import Faker

def generate_phone_numbers(num_numbers: int, output_file: str):
    fake = Faker()
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['phone'])
        for i in range(num_numbers):
            writer.writerow([fake.phone_number()])
            if (i + 1) % 1000000 == 0:
                print(f"{i + 1} phone numbers generated.")

if __name__ == '__main__':
    total_numbers = 20000000
    output_filename = 'phone_numbers.csv'
    generate_phone_numbers(total_numbers, output_filename)
