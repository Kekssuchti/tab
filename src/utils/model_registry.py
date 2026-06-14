from src.adapter.limix_adapter import LimixAdapter
from src.adapter.mitra_adapter import MitraAdapter
from src.adapter.orion_bix_adapter import OrionBixAdapter
from src.adapter.orion_msp_adapter import OrionMSPAdapter
from src.adapter.tabicl_adapter import TabICLAdapter
from src.adapter.tabpfn_adapter import TabPFNAdapter

MODEL_REGISTRY_CLS = {
    "tabpfn-3": TabPFNAdapter,
    "tabicl-2": TabICLAdapter,
    "limix-2m": LimixAdapter,
    "limix-16m": LimixAdapter,
    "mitra": MitraAdapter,
    "orion-msp": OrionMSPAdapter,
    "orion-bix": OrionBixAdapter,
}


MODEL_REGISTRY_REG = {
    "tabpfn-3": TabPFNAdapter,
    "tabicl-2": TabICLAdapter,
    "limix-2m": LimixAdapter,
    "limix-16m": LimixAdapter,
    "mitra": MitraAdapter,
}
