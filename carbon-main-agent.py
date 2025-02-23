import json
import boto3
from datetime import datetime
import uuid
import time

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')
sqs = boto3.client('sqs')

# Constants
TABLE_NAME = 'carbon-assistant-data'
QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/[your-account-id]/carbon-bedrock-queue'

def store_conversation(conversation_id, user_input, response):
    """Store conversation in DynamoDB with proper formatting"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        timestamp = str(int(datetime.now().timestamp() * 1000))
        
        item = {
            'conversationId': conversation_id,
            'timestamp': timestamp,
            'userInput': user_input,
            'response': response,
            'type': 'conversation'
        }
        
        table.put_item(Item=item)
    except Exception as e:
        print(f"DynamoDB error: {str(e)}")
        return None

def enqueue_request(prompt, conversation_id):
    """Add request to SQS queue when throttled"""
    try:
        message_body = {
            'prompt': prompt,
            'conversation_id': conversation_id,
            'timestamp': str(int(datetime.now().timestamp() * 1000))
        }
        
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageAttributes={
                'RequestType': {
                    'DataType': 'String',
                    'StringValue': 'bedrock_inference'
                }
            }
        )
        return True
    except Exception as e:
        print(f"SQS error: {str(e)}")
        return False

def invoke_bedrock_with_queue(prompt, conversation_id):
    """Try direct invocation, fall back to queue if throttled"""
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read().decode())
        return {
            'status': 'direct',
            'response': response_body['content'][0]['text']
        }
        
    except Exception as e:
        if "ThrottlingException" in str(e):
            if enqueue_request(prompt, conversation_id):
                return {
                    'status': 'queued',
                    'response': "Your request has been queued due to high demand. You'll receive a response shortly via our notification system. In the meantime, here's some general guidance: Carbon emissions from electricity usage depend on your region's energy mix and the time of consumption. Consider implementing energy efficiency measures and shifting usage to off-peak hours when possible."
                }
        raise

def invoke_calculation_agent(query):
    """Invoke the calculation Lambda function"""
    try:
        response = lambda_client.invoke(
            FunctionName='carbon-calculation-agent',
            InvocationType='RequestResponse',
            Payload=json.dumps({'query': query})
        )
        return json.loads(response['Payload'].read())
    except Exception as e:
        print(f"Calculation agent invocation error: {str(e)}")
        raise

def lambda_handler(event, context):
    try:
        # Extract input from event
        body = json.loads(event.get('body', '{}'))
        user_input = body.get('input', '')
        conversation_id = body.get('conversationId', str(uuid.uuid4()))
        
        if not user_input:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Input is required'})
            }
        
        # Process query with queue fallback
        if "carbon" in user_input.lower() or "emission" in user_input.lower():
            try:
                calc_results = invoke_calculation_agent(user_input)
                prompt = f"Explain these carbon emission calculation results briefly and clearly: {calc_results}"
            except Exception:
                prompt = f"Provide general guidance about carbon emissions for {user_input}. Keep it brief."
                
            result = invoke_bedrock_with_queue(prompt, conversation_id)
            
        else:
            result = invoke_bedrock_with_queue(user_input, conversation_id)
        
        # Store conversation if direct response
        if result['status'] == 'direct':
            store_conversation(conversation_id, user_input, result['response'])
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': result['response'],
                'conversationId': conversation_id,
                'status': result['status']
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
