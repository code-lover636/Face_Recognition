#CODE 12-04-2024 (1)
#QR DATA AND AADAR DATA INTEGRATED
import cv2
from pyzbar.pyzbar import decode
import psycopg2

# Function to scan and decode QR code
def scan_qr_code():
    # Capture video from default camera
    cap = cv2.VideoCapture(0)

    error_shown = False  # Flag to track if error has been shown

    while True:
        ret, frame = cap.read()

        # Decode QR code
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            # Check if the decoded data starts with "mdl"
            if obj.data.decode('utf-8').startswith("mdl"):
                data = obj.data.decode('utf-8')
                print("Decoded Data:", data)
                return data

            elif not error_shown:  # If error hasn't been shown yet
                print("Error: QR code without initial string 'mdl' scanned")
                error_shown = True

        cv2.imshow("QR Code Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to scan Aadhar number and retrieve user data
def scan_aadhar():
    # Taking Aadhar number input from the user
    aadhar_input = input("Enter Aadhar number: ")

    # Retrieving user data by Aadhar number
    user = get_user_by_aadhar(aadhar_input)
    if user:
        print("User found:")
        print("Name:", user[0])
        print("Aadhar Number:", user[1])
        print("Address:", user[2])
        return user
    else:
        print("User not found.")
        return None

# Function to insert data into PostgreSQL database along with Aadhar details
def insert_into_database(data, name, aadhar_number, address):
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname="demo",
            user="postgres",
            password="akkupakku",
            host="localhost",
            port="5432"
        )

        # Create a cursor object using the connection
        cursor = conn.cursor()

        # Execute SQL query to insert data into the table
        cursor.execute("INSERT INTO qrdb (data, name, aadhar_number, address) VALUES (%s, %s, %s, %s)", (data, name, aadhar_number, address))

        # Commit the transaction
        conn.commit()

        # Close cursor and connection
        cursor.close()
        conn.close()

        print("Data inserted into PostgreSQL database successfully")

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while inserting data into PostgreSQL database:", error)

# Function to retrieve user data based on Aadhar number
def get_user_by_aadhar(aadhar_number):
    conn = psycopg2.connect(
        dbname="demo",
        user="postgres",
        password="akkupakku",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    sql = "SELECT * FROM aadhardb WHERE aadhar_number = %s;"
    cur.execute(sql, (aadhar_number,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user

def check_qr_existence(qr_data):
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname="demo",
            user="postgres",
            password="akkupakku",
            host="localhost",
            port="5432"
        )

        # Create a cursor object using the connection
        cursor = conn.cursor()

        # Execute SQL query to check if QR code data exists
        cursor.execute("SELECT data FROM qrdb WHERE data = %s", (qr_data,))
        qr_data_exists = cursor.fetchone()

        # Close cursor and connection
        cursor.close()
        conn.close()

        return qr_data_exists is not None

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while checking QR code existence in PostgreSQL database:", error)
        return False
    
def check_qr_existence_with_aadhar(qr_data):
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname="demo",
            user="postgres",
            password="akkupakku",
            host="localhost",
            port="5432"
        )

        # Create a cursor object using the connection
        cursor = conn.cursor()

        # Execute SQL query to check if QR data exists with Aadhar details
        cursor.execute("SELECT data FROM qrdb WHERE data = %s AND aadhar_number IS NOT NULL", (qr_data,))
        qr_data_with_aadhar = cursor.fetchone()

        # Close cursor and connection
        cursor.close()
        conn.close()

        return qr_data_with_aadhar is not None

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while checking QR code existence with Aadhar details in PostgreSQL database:", error)
        return False
def get_aadhar_number_from_qr(qr_data):
    # Assuming Aadhar number is encoded as the last 12 characters of the QR data
    aadhar_number = qr_data[-12:]
    return aadhar_number

def scan_and_recognize():
    qr_data = scan_qr_code()
    if qr_data:
        if check_qr_existence(qr_data):  # If QR data exists in qrdb
            if check_qr_existence_with_aadhar(qr_data):  # If QR data exists in qrdb with Aadhar details
                aadhar_number = get_aadhar_number_from_qr(qr_data)
                aadhar_details = get_user_by_aadhar(aadhar_number)
                if aadhar_details:
                    print("Access granted for:", aadhar_details[0])  # Print name associated with Aadhar number
                else:
                    print("Access granted")
            else:  # If QR data exists in qrdb without Aadhar details
                aadhar_number = input("Enter Aadhar number: ")
                aadhar_details = get_user_by_aadhar(aadhar_number)
                if aadhar_details:
                    insert_into_database(qr_data, aadhar_details[0], aadhar_number, aadhar_details[1])
                    print("Access granted for:", aadhar_details[0])  # Print name associated with Aadhar number
                else:
                    print("Aadhar details not found.")
        else:  # If QR data doesn't exist in qrdb
            print("QR code data not found in the database.")
            
            # Prompt user to insert QR data into qrdb
            insert_qr_into_database = input("Do you want to insert the QR data into the database? (yes/no): ")
            if insert_qr_into_database.lower() == "yes":
                aadhar_number = input("Enter Aadhar number: ")
                aadhar_details = get_user_by_aadhar(aadhar_number)
                if aadhar_details:
                    insert_into_database(qr_data, aadhar_details[0], aadhar_number, aadhar_details[1])
                    print("QR data inserted into database and access granted for:", aadhar_details[0])
                else:
                    print("Aadhar details not found.")
            else:
                print("Access denied.")

# Call the function to scan both QR code and Aadhar number
scan_and_recognize()