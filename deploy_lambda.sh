#!/bin/bash
# ============================================================================
# Deploy 5 EMA Backtest Lambda Function
# ============================================================================
# Prerequisites:
#   - AWS CLI configured with credentials
#   - pip installed
#
# Usage:
#   chmod +x deploy_lambda.sh
#   ./deploy_lambda.sh
# ============================================================================

set -e

# --- Configuration ---
FUNCTION_NAME="5ema-backtest"
REGION="us-east-1"
S3_BUCKET="5ema-backtest-results"
RUNTIME="python3.11"
HANDLER="lambda_handler.lambda_handler"
TIMEOUT=900          # 15 minutes
MEMORY_SIZE=512      # 512 MB
ROLE_NAME="5ema-backtest-lambda-role"

# --- Paths ---
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/lambda_build"
PACKAGE_FILE="$PROJECT_DIR/lambda_package.zip"

echo "=== 5 EMA Backtest Lambda Deployment ==="
echo "Project: $PROJECT_DIR"
echo "Function: $FUNCTION_NAME"
echo "Region: $REGION"
echo ""

# --- Step 1: Create build directory ---
echo "[1/7] Preparing build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# --- Step 2: Install dependencies ---
echo "[2/7] Installing Python dependencies..."
pip install --target "$BUILD_DIR" \
    pandas numpy requests boto3 \
    --quiet --no-cache-dir

# --- Step 3: Copy source files ---
echo "[3/7] Copying source files..."
cp "$PROJECT_DIR/lambda_handler.py" "$BUILD_DIR/"
cp "$PROJECT_DIR/five_ema_strategy.py" "$BUILD_DIR/"

# --- Step 4: Create ZIP package ---
echo "[4/7] Creating deployment package..."
cd "$BUILD_DIR"
zip -r "$PACKAGE_FILE" . -q
cd "$PROJECT_DIR"
PACKAGE_SIZE=$(du -h "$PACKAGE_FILE" | cut -f1)
echo "  Package size: $PACKAGE_SIZE"

# --- Step 5: Create S3 bucket (if not exists) ---
echo "[5/7] Ensuring S3 bucket exists..."
if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "  Bucket $S3_BUCKET already exists"
else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION"
    echo "  Created bucket $S3_BUCKET"
fi

# --- Step 6: Create/Get IAM Role ---
echo "[6/7] Setting up IAM role..."
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ] || [ "$ROLE_ARN" = "None" ]; then
    echo "  Creating IAM role: $ROLE_NAME"

    # Trust policy for Lambda
    cat > /tmp/trust-policy.json << 'TRUST'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}
TRUST

    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --query 'Role.Arn' --output text)

    # Attach policies
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    # S3 access policy
    cat > /tmp/s3-policy.json << S3POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::$S3_BUCKET",
                "arn:aws:s3:::$S3_BUCKET/*"
            ]
        }
    ]
}
S3POLICY

    aws iam put-role-policy --role-name "$ROLE_NAME" \
        --policy-name "${ROLE_NAME}-s3-access" \
        --policy-document file:///tmp/s3-policy.json

    echo "  Role created: $ROLE_ARN"
    echo "  Waiting 10s for role propagation..."
    sleep 10
else
    echo "  Role exists: $ROLE_ARN"
fi

# --- Step 7: Create or Update Lambda Function ---
echo "[7/7] Deploying Lambda function..."

FUNCTION_EXISTS=$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null && echo "yes" || echo "no")

if [ "$FUNCTION_EXISTS" = "yes" ]; then
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$PACKAGE_FILE" \
        --region "$REGION" \
        --no-cli-pager

    # Wait for update to complete
    aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null || true

    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --environment "Variables={S3_BUCKET=$S3_BUCKET,CLICKHOUSE_HOST=34.200.220.45,CLICKHOUSE_PORT=8123,CLICKHOUSE_USER=default,CLICKHOUSE_PASSWORD=,CLICKHOUSE_DATABASE=tradelayout,TELEGRAM_CHAT_ID=6343050453}" \
        --region "$REGION" \
        --no-cli-pager
else
    echo "  Creating new function..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --handler "$HANDLER" \
        --role "$ROLE_ARN" \
        --zip-file "fileb://$PACKAGE_FILE" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --environment "Variables={S3_BUCKET=$S3_BUCKET,CLICKHOUSE_HOST=34.200.220.45,CLICKHOUSE_PORT=8123,CLICKHOUSE_USER=default,CLICKHOUSE_PASSWORD=,CLICKHOUSE_DATABASE=tradelayout,TELEGRAM_CHAT_ID=6343050453}" \
        --region "$REGION" \
        --no-cli-pager
fi

# --- Cleanup ---
echo ""
echo "Cleaning up build directory..."
rm -rf "$BUILD_DIR"

echo ""
echo "=== Deployment Complete ==="
echo "Function : $FUNCTION_NAME"
echo "Region   : $REGION"
echo "S3 Bucket: $S3_BUCKET"
echo "Package  : $PACKAGE_FILE"
echo ""
echo "Test with:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME \\"
echo "    --payload '{\"date\": \"2024-10-03\"}' \\"
echo "    --region $REGION output.json && cat output.json"
echo ""
echo "Run full backtest:"
echo "  python orchestrator.py --start 2024-06-01 --end 2024-12-31 --mode async"
