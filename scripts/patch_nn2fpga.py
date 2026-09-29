import os

repo_path = os.path.abspath("nn2fpga_repo")
board_util_path = os.path.join(repo_path, "nn2fpga", "compiler", "utils", "board_util.py")

if os.path.exists(board_util_path):
    with open(board_util_path, "r") as f:
        content = f.read()
    
    # Replace the hardcoded container path with the actual local repo path
    new_content = content.replace("/workspace/NN2FPGA", repo_path)
    
    with open(board_util_path, "w") as f:
        f.write(new_content)
    print("✅ Successfully patched board_util.py with local paths!")
else:
    print("❌ Could not find board_util.py at expected path.")
