import sqlite3
import seaborn as sns

def create_database(db_name="penguins.db"):
    """
    Creates a SQLite database with two tables:
      1. ISLANDS   (island_id, name)
      2. PENGUINS  (species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, island_id, animal_id)

    Populates them with data from the Seaborn penguins dataset.
    """
    # 1. Load and clean dataset
    penguins = sns.load_dataset("penguins").dropna()

    # 2. Create a unique ID for each island
    unique_islands = penguins["island"].unique()
    island_map = {island: idx + 1 for idx, island in enumerate(unique_islands)}

    # 3. Add island_id and a simple animal_id to the DataFrame
    penguins["island_id"] = penguins["island"].map(island_map)
    penguins["animal_id"] = penguins.index + 1

    # 4. Connect to SQLite and create tables
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # (Optional) Drop tables if you want a fresh start each time you run the script
    # cur.execute("DROP TABLE IF EXISTS PENGUINS;")
    # cur.execute("DROP TABLE IF EXISTS ISLANDS;")

    # Create ISLANDS table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ISLANDS (
            island_id INTEGER PRIMARY KEY,
            name TEXT
        );
        """
    )

    # Create PENGUINS table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS PENGUINS (
            species TEXT,
            bill_length_mm REAL,
            bill_depth_mm REAL,
            flipper_length_mm REAL,
            body_mass_g REAL,
            sex TEXT,
            island_id INTEGER,
            animal_id INTEGER,
            FOREIGN KEY (island_id) REFERENCES ISLANDS(island_id)
        );
        """
    )

    # 5. Insert data into ISLANDS table
    for island_name, island_id in island_map.items():
        cur.execute(
            """
            INSERT INTO ISLANDS (island_id, name)
            VALUES (?, ?);
            """,
            (island_id, island_name)
        )

    # 6. Insert data into PENGUINS table
    for _, row in penguins.iterrows():
        cur.execute(
            """
            INSERT INTO PENGUINS (
                species,
                bill_length_mm,
                bill_depth_mm,
                flipper_length_mm,
                body_mass_g,
                sex,
                island_id,
                animal_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row["species"],
                row["bill_length_mm"],
                row["bill_depth_mm"],
                row["flipper_length_mm"],
                row["body_mass_g"],
                row["sex"],
                row["island_id"],
                row["animal_id"]
            )
        )

    # 7. Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
