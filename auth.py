"""
Zerodha Kite Authentication Module
Handles login and access token management
"""

from kiteconnect import KiteConnect
import logging
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZerodhaAuth:
    def __init__(self, api_key, api_secret):
        """
        Initialize Zerodha authentication

        Args:
            api_key: Your Kite Connect API key
            api_secret: Your Kite Connect API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = None
        self.token_file = "access_token.json"

    def get_login_url(self):
        """Generate login URL for manual authentication"""
        login_url = self.kite.login_url()
        logger.info(f"Login URL: {login_url}")
        return login_url

    def generate_session(self, request_token):
        """
        Generate access token from request token

        Args:
            request_token: Request token from redirect URL after login

        Returns:
            dict: Session data including access token
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)

            # Save token with metadata
            self.save_token(data)
            logger.info("Access token generated and saved successfully")
            return data
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            raise

    def save_token(self, session_data):
        """Save access token to file"""
        token_data = {
            "access_token": session_data["access_token"],
            "user_id": session_data["user_id"],
            "user_name": session_data["user_name"],
            "generated_at": datetime.now().isoformat()
        }

        with open(self.token_file, 'w') as f:
            json.dump(token_data, f, indent=4)
        logger.info(f"Token saved to {self.token_file}")

    def load_token(self):
        """Load access token from file"""
        if not os.path.exists(self.token_file):
            logger.warning("Token file not found")
            return None

        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)

            self.access_token = token_data["access_token"]
            self.kite.set_access_token(self.access_token)
            logger.info("Access token loaded successfully")
            return token_data
        except Exception as e:
            logger.error(f"Error loading token: {e}")
            return None

    def get_kite_instance(self):
        """Return authenticated Kite instance"""
        if not self.access_token:
            raise Exception("No access token available. Please authenticate first.")
        return self.kite


def main():
    """Example usage"""
    # Replace with your actual credentials
    API_KEY = "pi2xk2iokf0u2nt3"
    API_SECRET = "ssifv93g7g6xo9lofpduwwzy4n45q3d2"

    auth = ZerodhaAuth(API_KEY, API_SECRET)

    # Try loading existing token
    token_data = auth.load_token()

    if not token_data:
        # Generate new token
        print("No existing token found. Please authenticate:")
        print(auth.get_login_url())
        print("\nAfter login, paste the request token from the redirect URL:")
        request_token = input("Request token: ").strip()
        auth.generate_session(request_token)
    else:
        print(f"Loaded existing token for user: {token_data['user_name']}")

    # Test connection
    try:
        kite = auth.get_kite_instance()
        profile = kite.profile()
        print(f"\nAuthenticated as: {profile['user_name']}")
        print(f"Email: {profile['email']}")
    except Exception as e:
        print(f"Authentication failed: {e}")


if __name__ == "__main__":
    main()