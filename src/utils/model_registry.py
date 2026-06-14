from src.adapter.tabicl_adapter import TabICLAdapter
from src.adapter.tabpfn_adapter import TabPFNAdapter

MODEL_REGISTRY = {
    "tabpfn-3": TabPFNAdapter,
    "tabicl-2": TabICLAdapter,
}
