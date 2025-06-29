import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Configuration manager for CS2 Portfolio Optimizer"""
    # Steam API Configuration
    STEAM_APP_ID: int = 730   # 730 for cs2/csgo
    STEAM_SESSIONID: str = os.getenv("SESSIONID", "")
    STEAM_LOGIN_SECURE: str = os.getenv("STEAMLOGINSECURE", "")
    
    # Data Processing Parameters
    LOW_PRICE_THRESHOLD: float = 0.06  # threshold for filtering
    MIN_DAYS_DIFF: int = 15  # Minimum days for price filtering
    SMOOTHING_WINDOW: int = 5  # Moving average window
    SPIKE_WINDOW: int = 12  # Window for spike detection
    SPIKE_DEVIATION_THRESHOLD: float = 0.2  # 20% deviation threshold
    SPIKE_REVERSION_WINDOW: int = 3  # Days to check for reversion
    
    # Portfolio Optimization Parameters
    RISK_FREE_RATE: float = 0.0
    DAYS_IN_SAMPLE: int = 365  # Days to annualize returns
    MIN_VOL_THRESHOLD: float = 1e-6  # Minimum volatility threshold
    EFFICIENT_FRONTIER_POINTS: int = 50  # Number of frontier points
    
    # Monte Carlo Simulation Parameters
    DEFAULT_SIMULATIONS: int = 100
    DEFAULT_TIMEFRAME: int = 365
    CONFIDENCE_LEVEL: float = 0.05  # 5% VaR
    
    # File Paths
    DATA_RAW_PATH: str = "data/raw/market_prices"
    DATA_PROCESSED_PATH: str = "data/processed"
    
    # Freq Mappings
    FREQ_TO_DAYS: Dict[str, int] = None
    
    def __post_init__(self):
        if self.FREQ_TO_DAYS is None:
            self.FREQ_TO_DAYS = {
                'D': 365,
                'W': 52,
                'M': 12,
                'Q': 4,
                '3M': 4,
                '6M': 2,
                'Y': 1,
                'B': 252
            }
    
    # Validation methods
    def validate_steam_credentials(self) -> bool:
        """Validate that Steam credentials are present"""
        return bool(self.STEAM_SESSIONID and self.STEAM_LOGIN_SECURE)
    
    def get_steam_cookies(self) -> Dict[str, str]:
        """Get Steam cookies for API requests"""
        if not self.validate_steam_credentials():
            raise ValueError("Steam credentials not configured (check .env file)")
        
        return {
            "sessionid": self.STEAM_SESSIONID,
            "steamLoginSecure": self.STEAM_LOGIN_SECURE
        }
    
    # Convenience methods
    def get_data_path(self, skin_type: str) -> str:
        """Get path for specific skin type data"""
        return os.path.join(self.DATA_RAW_PATH, skin_type)
    
    def get_processed_path(self, frequency: str = "D") -> str:
        """Get path for processed data"""
        return os.path.join(self.DATA_PROCESSED_PATH, frequency)

config = Config()