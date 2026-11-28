import fed_fraud_noniid.patch  # noqa: F401  (must be imported first)
from fed_fraud import client_app as _orig

app = _orig.app
