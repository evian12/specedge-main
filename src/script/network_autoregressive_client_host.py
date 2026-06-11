import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import log


SPECEDGE_ROOT = Path(__file__).absolute().parents[2]


def main(config_file: Path) -> None:
    with config_file.open() as file:
        config = yaml.safe_load(file)

    base = config["base"]
    client = config["client"]
    server = config["server"]
    log_dir = Path(base["result_path"]) / base["exp_name"]
    log.configure_logging(
        log.get_default_log_config(log_dir, "network_ar_client_host")
    )
    logger = log.get_logger()

    remote_port = int(client.get("tunnel_port", 18002))
    server_port = int(server.get("port", 8002))
    ssh_key = str(Path(base["ssh_key"]).expanduser())
    configured_clients = [
        (node_name, node_client)
        for node_name, clients in config["node"].items()
        for node_client in clients
    ]
    if len(configured_clients) != 1:
        raise ValueError(
            "The network autoregressive baseline currently supports "
            "exactly one client"
        )

    processes = []
    for client_idx, (node_name, _) in enumerate(configured_clients):
        env_vars = {
            "SPECEDGE_RESULT_PATH": base["result_path"],
            "SPECEDGE_EXP_NAME": base["exp_name"],
            "SPECEDGE_PROCESS_NAME": f"network_ar_client_{client_idx}",
            "SPECEDGE_HOST": f"127.0.0.1:{remote_port}",
            "SPECEDGE_TARGET_MODEL": server["target_model"],
            "SPECEDGE_DATASET": client["dataset"],
            "SPECEDGE_MAX_NEW_TOKENS": client["max_new_tokens"],
            "SPECEDGE_MAX_REQUEST_NUM": client["max_request_num"],
            "SPECEDGE_SAMPLE_REQ_CNT": client["sample_req_cnt"],
            "SPECEDGE_REQ_OFFSET": client["req_offset"],
            "SPECEDGE_CLIENT_IDX": client_idx,
            "SPECEDGE_REASONING": client.get("reasoning", False),
        }
        command = f"cd {SPECEDGE_ROOT} && "
        for key, value in env_vars.items():
            command += f'export {key}="{value}" && '
        command += "bash ./script/network_autoregressive_client.sh"

        logger.info(
            "Starting network autoregressive client_%d on %s",
            client_idx,
            node_name,
        )
        process = subprocess.Popen(  # noqa: S603
            [
                "ssh",
                "-i",
                ssh_key,
                "-o",
                "ExitOnForwardFailure=yes",
                "-R",
                f"{remote_port}:127.0.0.1:{server_port}",
                node_name,
                command,
            ],  # noqa: S607
            stdout=None,
            stderr=sys.stderr.buffer,
            text=True,
        )
        processes.append(process)

    for client_idx, process in enumerate(processes):
        process.wait()
        logger.info("network_ar_client_%d finished", client_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    main(args.config)
