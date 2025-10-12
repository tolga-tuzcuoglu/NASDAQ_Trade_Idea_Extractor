# 🚨 URGENT: API Key Rotation Instructions

## CRITICAL SECURITY ALERT

**A Google API key was exposed in the repository and has been revoked.**
**⚠️ The exposed key should not be used and has been deactivated.**

## IMMEDIATE ACTIONS REQUIRED:

### 1. 🔴 ROTATE THE EXPOSED API KEY

#### Step 1: Generate New API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the new API key immediately

#### Step 2: Revoke Old API Key
1. In Google AI Studio, find the old key (the one that was exposed)
2. Click "Delete" or "Revoke" next to the old key
3. Confirm the deletion

#### Step 3: Update Local Environment
1. Open your `.env` file
2. Replace the old key with the new one:
   ```
   GEMINI_API_KEY=your_new_api_key_here
   ```
3. Save the file

### 2. 🔴 MONITOR FOR UNAUTHORIZED USAGE

#### Check API Usage:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Check the "Usage" tab for the old key
3. Look for unusual activity or high usage
4. If suspicious activity is found, report it immediately

#### Monitor for:
- Unusual API call patterns
- High usage volumes
- Calls from unknown IP addresses
- Requests outside normal business hours

### 3. 🔴 VERIFY SECURITY MEASURES

#### Test the New Setup:
1. Run the security check: `python security_check.py`
2. Test the application with the new API key
3. Verify no old keys are still in use

#### Check Repository:
1. Ensure no `.env` files are tracked in git
2. Verify `.gitignore` includes all sensitive patterns
3. Run `git status` to confirm no sensitive files are staged

## PREVENTION MEASURES:

### 1. Pre-commit Security Check
```bash
# Run before every commit
python security_check.py
```

### 2. Environment Variable Best Practices
- Never hardcode API keys in source code
- Always use environment variables
- Use `.env` files for local development only
- Never commit `.env` files to git

### 3. Regular Security Audits
- Run security checks before each commit
- Review repository for exposed secrets monthly
- Monitor API usage regularly
- Keep security documentation updated

## EMERGENCY CONTACTS:

If you discover unauthorized usage:
1. **Immediately** revoke all API keys
2. **Contact** Google AI Studio support
3. **Review** all recent API usage
4. **Update** security measures

## STATUS CHECKLIST:

- [ ] New API key generated
- [ ] Old API key revoked
- [ ] Local `.env` file updated
- [ ] Application tested with new key
- [ ] Security check passed
- [ ] No sensitive data in repository
- [ ] API usage monitored

**This is a CRITICAL security issue. Complete these steps immediately.**
