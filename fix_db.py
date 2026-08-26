import sqlite3
import os
import shutil

src_db = 'chroma_db_old/chroma.sqlite3'
dst_db = 'chroma_db/chroma.sqlite3'

# Create chroma_db if it doesn't exist
os.makedirs('chroma_db', exist_ok=True)

# Remove the destination if it exists
if os.path.exists(dst_db):
    os.remove(dst_db)

print(f"Connecting to {src_db}...")
try:
    src_conn = sqlite3.connect(src_db, timeout=10)
    dst_conn = sqlite3.connect(dst_db)
    
    print("Backing up database...")
    src_conn.backup(dst_conn)
    
    src_conn.close()
    dst_conn.close()
    print("Backup completed successfully.")
except Exception as e:
    print(f"Backup failed: {e}")
    print("Attempting shutil.copy2 as fallback...")
    shutil.copy2(src_db, dst_db)
    print("Copy completed.")

# Also need to copy the f70a4ea1-6d1f-4baf-b8c8-0b0e4cf26a2f folder
src_uuid_folder = 'chroma_db_old/f70a4ea1-6d1f-4baf-b8c8-0b0e4cf26a2f'
dst_uuid_folder = 'chroma_db/f70a4ea1-6d1f-4baf-b8c8-0b0e4cf26a2f'

if os.path.exists(src_uuid_folder):
    if os.path.exists(dst_uuid_folder):
        shutil.rmtree(dst_uuid_folder)
    shutil.copytree(src_uuid_folder, dst_uuid_folder)
    print("UUID folder copied.")

# Remove macOS extended attributes using os.system
os.system('xattr -rc chroma_db/')
print("Cleared extended attributes.")
