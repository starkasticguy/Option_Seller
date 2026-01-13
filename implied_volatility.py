import pandas as pd
import numpy as np
from scipy.optimize import fmin
import scipy.stats as si
from datetime import datetime

# Load data
df = pd.read_csv("option_chain_data.csv")
print("Original Data:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Convert expiry to datetime if needed
df['expiry'] = pd.to_datetime(df['expiry'])

# Risk-free rate
risk_free_rate = 0.0525


def NORMSDIST(x):
    """Standard normal cumulative distribution function"""
    return si.norm.cdf(x, 0.0, 1.0)


def ImpliedVolatilityPut(s, S, X, r, T, pm):
    """
    Calculate Black-Scholes put price and return squared error

    Args:
        s: volatility (as array [sigma])
        S: spot price
        X: strike price
        r: risk-free rate
        T: time to maturity (in years)
        pm: market price of put
    """
    sigma = s[0]

    # Avoid division by zero
    if sigma <= 0 or T <= 0:
        return 1e10

    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / X) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # Black-Scholes put price
    put_price = X * np.exp(-r * T) * NORMSDIST(-d2) - S * NORMSDIST(-d1)

    # Squared error
    val = (put_price - pm) ** 2

    return val


def ImpliedVolatilityCall(s, S, X, r, T, cm):
    """
    Calculate Black-Scholes call price and return squared error

    Args:
        s: volatility (as array [sigma])
        S: spot price
        X: strike price
        r: risk-free rate
        T: time to maturity (in years)
        cm: market price of call
    """
    sigma = s[0]

    # Avoid division by zero
    if sigma <= 0 or T <= 0:
        return 1e10

    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / X) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # Black-Scholes call price
    call_price = S * NORMSDIST(d1) - X * np.exp(-r * T) * NORMSDIST(d2)

    # Squared error
    val = (call_price - cm) ** 2

    return val


def calculate_time_to_maturity(expiry_date, current_date=None):
    """
    Calculate time to maturity in years

    Args:
        expiry_date: expiry date (datetime)
        current_date: current date (datetime), defaults to now
    """
    if current_date is None:
        current_date = datetime.now()

    days_to_expiry = (expiry_date - current_date).days

    # Convert to years (assuming 365 days)
    time_to_maturity = max(days_to_expiry / 365.0, 1 / 365.0)  # Minimum 1 day

    return time_to_maturity


def calculate_iv_for_row(row):
    """
    Calculate implied volatility for a single option row
    """
    try:
        S = row['spot_price']
        X = row['strike']
        r = risk_free_rate
        T = row['time_to_maturity']
        option_type = row['option_type']
        market_price = row['last_price']

        # Skip if market price is zero or very small
        if market_price < 0.01:
            return np.nan

        # Initial guess for volatility
        initial_guess = 0.14

        # Optimize based on option type
        if option_type == 'PE':
            result = fmin(
                ImpliedVolatilityPut,
                [initial_guess],
                args=(S, X, r, T, market_price),
                disp=False,
                ftol=1e-6,
                xtol=1e-6
            )
        elif option_type == 'CE':
            result = fmin(
                ImpliedVolatilityCall,
                [initial_guess],
                args=(S, X, r, T, market_price),
                disp=False,
                ftol=1e-6,
                xtol=1e-6
            )
        else:
            return np.nan

        implied_vol = result[0]

        # Sanity check: IV should be between 0.01 and 5 (1% to 500%)
        if implied_vol < 0.01 or implied_vol > 5:
            return np.nan

        return implied_vol

    except Exception as e:
        print(f"Error calculating IV for strike {row['strike']}, type {row['option_type']}: {e}")
        return np.nan


def calculate_moneyness(row):
    """
    Calculate moneyness (S/K for calls, K/S for puts)
    """
    S = row['spot_price']
    K = row['strike']
    option_type = row['option_type']

    if option_type == 'CE':
        # For calls: moneyness = S/K
        # > 1 means ITM, < 1 means OTM
        return S / K
    else:  # PE
        # For puts: moneyness = K/S
        # > 1 means ITM, < 1 means OTM
        return K / S


# Calculate time to maturity for each row
print("\nCalculating time to maturity...")
df['time_to_maturity'] = df['expiry'].apply(calculate_time_to_maturity)

# Calculate moneyness
print("Calculating moneyness...")
df['moneyness'] = df.apply(calculate_moneyness, axis=1)


# Add moneyness category
def categorize_moneyness(moneyness):
    if moneyness > 1.02:
        return 'ITM'
    elif moneyness < 0.98:
        return 'OTM'
    else:
        return 'ATM'


df['moneyness_category'] = df['moneyness'].apply(categorize_moneyness)

# Calculate implied volatility for each option
print("Calculating implied volatility (this may take a moment)...")
df['implied_volatility'] = df.apply(calculate_iv_for_row, axis=1)

# Convert IV to percentage for easier reading
df['implied_volatility_pct'] = df['implied_volatility'] * 100

# Display results
print("\n" + "=" * 80)
print("RESULTS - Options with Implied Volatility")
print("=" * 80)

# Show summary statistics
print(f"\nTotal options: {len(df)}")
print(f"Successfully calculated IV for: {df['implied_volatility'].notna().sum()} options")
print(f"Failed calculations: {df['implied_volatility'].isna().sum()} options")

# Show IV statistics
print("\nImplied Volatility Statistics:")
print(df['implied_volatility_pct'].describe())

# Display sample results
print("\n=== Sample Call Options (CE) ===")
ce_df = df[df['option_type'] == 'CE'].sort_values('strike')
print(ce_df[['strike', 'last_price', 'moneyness', 'moneyness_category',
             'implied_volatility_pct', 'time_to_maturity']].head(10))

print("\n=== Sample Put Options (PE) ===")
pe_df = df[df['option_type'] == 'PE'].sort_values('strike')
print(pe_df[['strike', 'last_price', 'moneyness', 'moneyness_category',
             'implied_volatility_pct', 'time_to_maturity']].head(10))

# Show volatility smile/skew
print("\n=== IV by Moneyness (ATM options) ===")
atm_options = df[df['moneyness_category'] == 'ATM']
if len(atm_options) > 0:
    print(atm_options[['strike', 'option_type', 'last_price', 'moneyness',
                       'implied_volatility_pct']].sort_values('strike'))
else:
    print("No ATM options found")

# Save enhanced dataframe
output_file = 'option_chain_with_iv.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Enhanced data saved to {output_file}")

# Create summary by strike for volatility smile analysis
print("\n=== Volatility Smile Summary ===")
smile_df = df.groupby(['strike', 'option_type'])['implied_volatility_pct'].mean().reset_index()
smile_pivot = smile_df.pivot(index='strike', columns='option_type', values='implied_volatility_pct')
print(smile_pivot.head(10))

print("\n" + "=" * 80)