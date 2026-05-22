import mysql.connector


# ---------------------------------------------------
# MYSQL CONNECTION
# ---------------------------------------------------

def connect_mysql():

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="rag_project"
    )

    return connection


# ---------------------------------------------------
# INSERT RAG DATA
# ---------------------------------------------------

def save_rag_data(question, answer):

    connection = connect_mysql()

    cursor = connection.cursor()

    query = """
    INSERT INTO rag_logs (
        question,
        answer
    )
    VALUES (%s, %s)
    """

    values = (
        question,
        answer
    )

    cursor.execute(
        query,
        values
    )

    connection.commit()

    cursor.close()

    connection.close()


# ---------------------------------------------------
# SELECT DATA
# ---------------------------------------------------

def fetch_rag_data():

    connection = connect_mysql()

    cursor = connection.cursor()

    query = """
    SELECT question, answer
    FROM rag_logs
    """

    cursor.execute(query)

    rows = cursor.fetchall()

    print("\nRAG PROJECT DATABASE RECORDS:\n")

    for row in rows:

        print(f"Question: {row[0]}")

        print(f"Answer: {row[1]}")

        print("-" * 50)

    cursor.close()

    connection.close()