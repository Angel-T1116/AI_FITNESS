import pickle
from pathlib import Path
# ← import Hasher from its new location:
from streamlit_authenticator.utilities.hasher import Hasher  # :contentReference[oaicite:0]{index=0}

# your user definitions
names = ["Angel", "Trujillo"]
usernames = ["angel", "trujillo"]
passwords = ["yourpassword1", "yourpassword2"]  # replace with your real plaintext passwords

# hash them
hashed_passwords = Hasher.hash_list(passwords)

# write out the pickle file
file_path = Path(__file__).parent / "hashed.pkl"
with open(file_path, "wb") as f:
    pickle.dump(hashed_passwords, f)
