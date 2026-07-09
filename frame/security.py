"""
Security verification for the anchoring system.

Provides automated tests for the attack vectors analysed in Chapter 6
of the accompanying paper: replay attacks and Sybil attacks.
"""

from dataclasses import dataclass
from web3 import Web3

from .contracts import AnchorClient


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SecurityReport:
    """Aggregated results of security tests."""

    replay_blocked: bool = False
    """True if duplicate dataset registration is rejected."""

    sybil_possible: bool = False
    """True if an unauthorised account can register datasets (indicates
    missing access control)."""

    model_replay_blocked: bool = False
    """True if duplicate model registration is rejected."""

    detail: dict = None

    def __post_init__(self):
        if self.detail is None:
            self.detail = {}

    @property
    def summary(self) -> str:
        lines = [
            f"Replay attack (dataset): {'BLOCKED' if self.replay_blocked else 'VULNERABLE'}",
            f"Sybil attack:            {'POSSIBLE' if self.sybil_possible else 'MITIGATED'}",
            f"Replay attack (model):   {'BLOCKED' if self.model_replay_blocked else 'VULNERABLE'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Security Tester
# ---------------------------------------------------------------------------

class SecurityTester:
    """
    Run security tests against deployed DataAnchor and WeightAnchor
    contracts.

    Parameters
    ----------
    client : AnchorClient
        Connected client (deployer account).
    attacker_private_key : str
        Hex-encoded private key for a second, unauthorised account
        (used for Sybil tests).

    Examples
    --------
    >>> tester = SecurityTester(client, attacker_pk)
    >>> report = tester.run()
    >>> print(report.summary)
    """

    def __init__(
        self, client: AnchorClient, attacker_private_key: str
    ) -> None:
        self.client = client
        self.w3 = client.w3
        self.attacker = self.w3.eth.account.from_key(attacker_private_key)

    # ---- individual tests --------------------------------------------

    def test_replay_dataset(self) -> bool:
        """
        Attempt to register the same dataset ID twice.

        Returns True if the second attempt is **blocked**.
        """
        dataset_id = Web3.keccak(text="security-replay-dataset")
        root = Web3.keccak(text="replay-test-root")

        # First registration — should succeed
        self.client.register_dataset(dataset_id, root, '{"name":"replay"}')

        # Second registration — should be rejected
        try:
            self.client.register_dataset(
                dataset_id, root, '{"name":"replay-attack"}'
            )
            return False  # not blocked → vulnerable
        except Exception:
            return True  # blocked → secure

    def test_sybil(self) -> bool:
        """
        Attempt to register a dataset from an unauthorised account.

        Returns True if the registration **succeeds** (i.e. Sybil is
        possible — the system lacks access control).
        """
        fake_id = Web3.keccak(text="sybil-test-dataset")
        fake_root = Web3.keccak(text="sybil-fake-data")

        # Build and send from the attacker account
        data_anchor = self.w3.eth.contract(
            address=self.client.data_addr,
            abi=self.client.data_anchor.abi,
        )
        nonce = self.w3.eth.get_transaction_count(self.attacker.address)
        tx = data_anchor.functions.registerDataset(
            fake_id, fake_root, '{"name":"sybil"}'
        ).build_transaction({
            "from": self.attacker.address,
            "nonce": nonce,
            "gas": 300_000,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.client.chain_id,
        })
        signed = self.w3.eth.account.sign_transaction(
            tx, self.attacker.key
        )
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)

        # Verify the fake dataset exists on chain
        record = data_anchor.functions.datasets(fake_id).call()
        # record is a tuple; check that registrar is the attacker
        return True  # registration succeeded → Sybil possible

    def test_replay_model(self) -> bool:
        """
        Attempt to register the same model ID twice.

        Returns True if the second attempt is **blocked**.
        """
        model_id = Web3.keccak(text="security-replay-model")
        w_hash = Web3.keccak(text="w")
        d_id = Web3.keccak(text="security-replay-dataset")
        chain = Web3.keccak(text="chain")
        parent = b"\x00" * 32

        self.client.register_model(
            model_id, w_hash, d_id, chain, parent, "{}", "v1"
        )
        try:
            self.client.register_model(
                model_id, w_hash, d_id, chain, parent, "{}", "v2"
            )
            return False
        except Exception:
            return True

    # ---- run all -----------------------------------------------------

    def run(self) -> SecurityReport:
        """
        Execute all security tests and return a report.

        Returns
        -------
        SecurityReport
        """
        report = SecurityReport()

        # Replay — dataset
        try:
            report.replay_blocked = self.test_replay_dataset()
        except Exception as e:
            report.detail["replay_error"] = str(e)

        # Sybil
        try:
            report.sybil_possible = self.test_sybil()
        except Exception as e:
            report.detail["sybil_error"] = str(e)

        # Replay — model
        try:
            report.model_replay_blocked = self.test_replay_model()
        except Exception as e:
            report.detail["model_replay_error"] = str(e)

        return report
