import paramiko
from scp import SCPClient
import os
import time

HOST = "10.0.0.174"
USER = "user"
PASS = "0000"
SUDO_PASS = "0000"

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def run_sudo_cmd(client, cmd):
    print(f"Running: {cmd}")
    stdin, stdout, stderr = client.exec_command(f"sudo -S {cmd}")
    stdin.write(SUDO_PASS + "\n")
    stdin.flush()
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if exit_status != 0:
        print(f"Error ({exit_status}): {err}")
    else:
        print(f"Success: {out[:200]}...")
    return exit_status, out, err

client = create_ssh_client(HOST, 22, USER, PASS)

# 1. Install PostgreSQL
run_sudo_cmd(client, "apt-get update")
run_sudo_cmd(client, "apt-get install -y postgresql postgresql-contrib")

# 2. Configure DB and User
print("Configuring Database...")
# Use psql via sudo -u postgres
cmds = [
    "sudo -S -u postgres psql -c \"CREATE DATABASE \\\"QuantumBot\\\";\"",
    # user postgres already exists but we might need to set password
    "sudo -S -u postgres psql -c \"ALTER USER postgres WITH PASSWORD 'Xvw6r9gc';\"",
]
for cmd in cmds:
    stdin, stdout, stderr = client.exec_command(cmd)
    stdin.write(SUDO_PASS + "\n")
    stdin.flush()
    print(stdout.read().decode())
    print(stderr.read().decode())

# 3. Configure pg_hba.conf and postgresql.conf to allow Docker connections
print("Configuring Network access...")
run_sudo_cmd(client, "sed -i \"s/#listen_addresses = 'localhost'/listen_addresses = '*'/\" /etc/postgresql/*/main/postgresql.conf")
run_sudo_cmd(client, "bash -c 'echo \"host all all 172.17.0.0/16 md5\" >> /etc/postgresql/*/main/pg_hba.conf'")
run_sudo_cmd(client, "bash -c 'echo \"host all all 10.0.0.0/8 md5\" >> /etc/postgresql/*/main/pg_hba.conf'")
run_sudo_cmd(client, "systemctl restart postgresql")

# 4. Create directory and copy files
run_sudo_cmd(client, "mkdir -p /opt/quantumbot")
run_sudo_cmd(client, "chown -R user:user /opt/quantumbot")

print("Copying files...")
with SCPClient(client.get_transport()) as scp:
    scp.put('.env', '/opt/quantumbot/.env')
    scp.put('api/QuantumBotDB-Prod.sql', '/opt/quantumbot/schema.sql')

# 5. Run Schema
print("Executing Schema...")
stdin, stdout, stderr = client.exec_command("sudo -S -u postgres psql -d QuantumBot -f /opt/quantumbot/schema.sql")
stdin.write(SUDO_PASS + "\n")
stdin.flush()
print(stdout.read().decode())
print(stderr.read().decode())

# 6. Start Watchtower
print("Starting Watchtower...")
run_sudo_cmd(client, "docker run -d --name watchtower -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower -i 30")

client.close()
print("Setup complete.")
