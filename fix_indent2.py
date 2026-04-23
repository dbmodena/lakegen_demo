with open("src/main.py", "r") as f:
    lines = f.readlines()

# Line 284 is 0-indexed 283
lines[283] = "        finally:\n"
lines[284] = "            # Clean up the temporary BLEND index\n"
lines[285] = "            if hasattr(self, 'blend_db_path') and self.blend_db_path.exists():\n"
lines[286] = "                try:\n"
lines[287] = "                    os.remove(self.blend_db_path)\n"
lines[288] = "                    print(f\"    🗑️  Temporary BLEND index removed.\")\n"
lines[289] = "                except Exception as e:\n"
lines[290] = "                    print(f\"    [!] Could not remove temp BLEND index: {e}\")\n"

with open("src/main.py", "w") as f:
    f.writelines(lines)
