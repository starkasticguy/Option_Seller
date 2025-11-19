# Zerodha Authentication Guide

This guide explains how to authenticate with Zerodha using the manual OAuth login flow (no Selenium required).

## Overview

The system uses **manual OAuth authentication** as recommended by the Kite API documentation. You will:
1. Get a login URL from the system
2. Manually log in through your browser
3. Copy the request token from the redirect URL
4. Paste it back into the system

**No passwords or credentials are stored** - you log in directly on Zerodha's secure website.

## Prerequisites

1. Zerodha trading account
2. API credentials from [Kite Developer Console](https://developers.kite.trade/)

## Setup (One-Time)

### Step 1: Create Kite App

1. Visit https://developers.kite.trade/
2. Log in with your Zerodha credentials
3. Click "Create New App"
4. Fill in the details:
   - **App Name**: BankNifty Trader (or any name you prefer)
   - **Redirect URL**: `http://127.0.0.1`
   - **Description**: Options trading system
5. Click "Create"

### Step 2: Get API Credentials

After creating the app, you'll receive:
- **API Key**: A string like `xxxxxxxxxxxxx`
- **API Secret**: A string like `yyyyyyyyyyyyyyyy`

**Keep these secret!** Never commit them to git.

### Step 3: Configure Environment

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```env
   ZERODHA_API_KEY=your_api_key_here
   ZERODHA_API_SECRET=your_api_secret_here
   ```

## First-Time Login

When you run the system for the first time, follow these steps:

### Step 1: Run the System

```bash
python run.py
```

or

```bash
python -m banknifty_trader.executor
```

### Step 2: Get Login URL

The system will display something like:

```
============================================================
ZERODHA LOGIN REQUIRED - MANUAL AUTHENTICATION
============================================================

Step 1: Copy this URL and open it in your browser:
https://kite.zerodha.com/connect/login?api_key=xxxxxxxxxxxxx

Step 2: Log in with your Zerodha credentials
        - Enter User ID and Password
        - Enter 2FA/TOTP if enabled

Step 3: After successful login, you'll be redirected to:
        127.0.0.1/?request_token=XXXXXXXXX&action=login&status=success

Step 4: Copy the 'request_token' value from the URL
        (everything after 'request_token=' and before '&')

Paste the request token here and press Enter:
>>>
```

### Step 3: Login in Browser

1. **Copy** the URL displayed in Step 1
2. **Paste** it into your browser
3. You'll see the Zerodha login page
4. **Enter** your User ID and Password
5. **Enter** your 2FA code (if enabled)
6. Click "Login"

### Step 4: Get Request Token

After successful login, the browser will try to redirect to:
```
http://127.0.0.1/?request_token=ABCD1234567890EFGH&action=login&status=success
```

The page may show "Unable to connect" - **this is normal!**

Copy the `request_token` value from the URL. In the example above, it's:
```
ABCD1234567890EFGH
```

### Step 5: Paste Request Token

Go back to the terminal where the system is running and paste the request token:

```
>>> ABCD1234567890EFGH
```

Press Enter.

### Step 6: Save Access Token

The system will generate and display an access token:

```
✓ Login successful!

============================================================
IMPORTANT: Save this access token for future use
============================================================
Access Token: xyz123abc456def789ghi012jkl345mno678pqr901stu234vwx567

Add this to your .env file:
ZERODHA_ACCESS_TOKEN=xyz123abc456def789ghi012jkl345mno678pqr901stu234vwx567
============================================================
```

**Copy this access token** and add it to your `.env` file:

```env
ZERODHA_ACCESS_TOKEN=xyz123abc456def789ghi012jkl345mno678pqr901stu234vwx567
```

## Subsequent Logins

Once you've saved the access token in your `.env` file:
- The system will use it automatically
- No manual login required
- **BUT**: Access tokens expire daily at midnight IST

## Daily Token Refresh

Zerodha access tokens are valid until **midnight IST** (Indian Standard Time).

### Option 1: Manual Refresh (Recommended)

Every day before trading, generate a fresh token:

1. Remove or comment out `ZERODHA_ACCESS_TOKEN` in `.env`
2. Run the system
3. Follow the manual login flow
4. Save the new access token

### Option 2: Auto-Refresh on Expiry

If you forget to refresh, the system will detect an expired token and prompt you to login again automatically.

## Security Best Practices

### DO:
✅ Keep your API key and secret secure
✅ Use `.env` file for credentials (it's in `.gitignore`)
✅ Generate fresh access tokens daily
✅ Log out of browser after getting request token
✅ Use 2FA/TOTP on your Zerodha account

### DON'T:
❌ Share your API key or secret with anyone
❌ Commit `.env` file to git
❌ Store passwords in files (not needed for this flow)
❌ Reuse old/expired access tokens
❌ Share access tokens

## Troubleshooting

### "Invalid API key"
- Check that `ZERODHA_API_KEY` in `.env` matches your Kite app's API key
- Ensure there are no extra spaces or quotes

### "Invalid API secret"
- Check that `ZERODHA_API_SECRET` in `.env` is correct
- Copy it exactly from the Kite developer console

### "Invalid request token"
- Request tokens expire after a few minutes
- Generate a fresh login URL by restarting the system
- Complete the login process quickly

### "Token expired" or "TokenException"
- Access tokens expire at midnight IST
- Remove old token from `.env`
- Follow the manual login flow again

### "Incorrect redirect_url"
- Make sure your Kite app's redirect URL is set to `http://127.0.0.1`
- No trailing slash or port number

### Browser shows "Unable to connect" after login
- This is **normal** - the redirect won't load a page
- Just copy the request token from the URL bar

## API Rate Limits

Zerodha has rate limits:
- **10 requests per second** per API key
- **1 login session generation** per second

The system handles rate limiting automatically.

## Why Manual Login?

### Advantages:
✅ **Secure**: You log in directly on Zerodha's website
✅ **Official**: Recommended by Kite API documentation
✅ **Simple**: No browser automation needed
✅ **Reliable**: No Selenium dependencies or browser driver issues
✅ **2FA Support**: Works with TOTP/authenticator apps

### Disadvantages:
❌ Requires manual action once per day
❌ Not fully automated

## Alternative: Selenium (Not Recommended)

While it's possible to automate login with Selenium, **we don't recommend it** because:
- Zerodha's terms of service discourage automation
- Fragile - breaks when UI changes
- Security concerns - storing passwords
- 2FA complications
- Extra dependencies

The manual flow takes ~30 seconds per day and is more reliable.

## Reference

- [Kite Connect Documentation](https://kite.trade/docs/connect/v3/)
- [API Authentication Flow](https://kite.trade/docs/connect/v3/user/#login-flow)
- [Developer Console](https://developers.kite.trade/)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the Kite API documentation
3. Check your API credentials in the developer console
4. Ensure your Zerodha account is active

---

**Remember**: Access tokens expire at midnight IST daily. Refresh them before trading!
