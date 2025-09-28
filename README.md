# Solady Vanity (CUDA)

A CUDA-accelerated vanity address generator for Solady's CREATE3. This tool searches for salts that make a deployed contract address match a desired prefix. Results stream to the console and are automatically appended to `results.txt`.


## Requirements

- Vast.ai account (for renting NVIDIA GPU instances).
- SSH key configured in Vast.ai.
- Basic Linux command line knowledge.

## Step-by-Step Instructions

### 1. Rent a GPU VM

1. Log in to [Vast.ai](https://vast.ai/).
2. Navigate to **Templates → NVIDIA CUDA → ▶ Play**.
3. Choose an Ubuntu 22.04 CUDA image with a GPU attached.
4. Rent the instance.

### 2. Connect to the VM

1. In Vast.ai, add your SSH public key (`~/.ssh/id_ed25519.pub`).
2. Copy the SSH command from the instance page (note the custom port).
3. Connect with:

   ```bash
   ssh -p <PORT> root@<INSTANCE_IP>
   ```

### 3. Verify GPU and CUDA

Run the following commands to confirm CUDA support:

```bash
nvidia-smi
nvcc --version || echo "nvcc not in PATH"
```

If `nvcc` is missing, install the CUDA toolkit or recreate the VM using the CUDA template.

### 4. Install Dependencies

```bash
apt-get update
apt-get install -y git build-essential cmake screen
```

### 5. Clone the Repository

```bash
git clone https://github.com/hodlthedoor/solady-vanity.git
cd solady-vanity
chmod +x run.sh
```

### 6. Run the Vanity Search

Replace `--prefix` with your desired hex prefix. Use `0x0000000000000000000000000000000000000000` as the deployer (placeholder). By default the tool uses Solady's CREATE3 init code hash; pass `--init-hash` if you need to override it for a custom contract.

```bash
./run.sh \
  --deployer 0x0000000000000000000000000000000000000000 \
  --prefix d0000000
```

The script builds the CUDA binary and starts searching. Output shows hash rate, estimated time to match, and successful salts. Each hit is displayed and appended to `results.txt` automatically.

> _Default init code hash_: `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f` (Solady CREATE3). Provide `--init-hash` if your target contract differs.

### 7. Run Persistently

To keep the search running in the background, use `screen`:

```bash
screen -S grind
```

Execute your vanity command inside the session. Detach with <kbd>Ctrl</kbd> + <kbd>A</kbd>, then <kbd>D</kbd>. Reattach later with:

```bash
screen -ls
screen -r grind
```

### 8. Monitor and Save Output

Tail the output log to monitor progress:

```bash
tail -f results.txt
```

## Benchmarks

| GPU        | Hashrate |
|------------|----------|
| RTX 4090   | 2.4 GH/s |
| RTX 5090   | 3.2 GH/s |

## References

- Solady CREATE3 contract: <https://github.com/Vectorized/solady/blob/main/src/utils/CREATE3.sol>
- Solady Documentation: <https://solady.org/>
