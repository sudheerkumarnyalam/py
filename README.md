Basic text to sql bot for select commands
Steps below :

	1.	Clone the repository by running:
git clone https://github.com/yourusername/nl-sql-bot.git
Then go into the project folder:
cd nl-sql-bot
	2.	(Optional) Create a Python virtual environment to isolate dependencies:
On macOS/Linux run: python3 -m venv venv
Activate it with: source venv/bin/activate
On Windows run: python -m venv venv
Activate it with: venv\Scripts\activate
	3.	Install the required Python packages:
pip install torch transformers tabulate
If you are on an Apple Silicon Mac (M1/M2), install torch using:
pip install torch torchvision torchaudio –extra-index-url https://download.pytorch.org/whl/cpu
	4.	Prepare a SQLite database file. For example, start SQLite shell with:
sqlite3 mydata.db
Inside the shell, create a table and insert sample data by typing:
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
INSERT INTO customers (name, country) VALUES (‘Alice’, ‘Germany’), (‘Bob’, ‘USA’);
Exit the shell with: .quit
	5.	Run the bot with:
python nl_sql_bot.py –db mydata.db
	6.	On the first run, the script will automatically download the Microsoft Phi-2 model (~4.7GB) from Hugging Face and cache it locally (usually in ~/.cache/huggingface/hub/). This step requires an internet connection and may take some time depending on your bandwidth.
	7.	After the model is downloaded, you can start typing natural language queries such as:
List all customers from Germany
The bot will generate the appropriate SQL, execute it on your SQLite database, and display the results.

⸻

If you want to avoid downloading the Phi model during the bot execution, you can pre-download it manually using the Hugging Face CLI tool:
Install it by running: pip install huggingface_hub
Log in (if needed) with: huggingface-cli login
Download the model repository with: huggingface-cli repo clone microsoft/phi-2
