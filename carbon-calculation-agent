import json
import boto3
from datetime import datetime
import uuid
import time

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

TABLE_NAME = 'carbon-assistant-data'

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
        raise

def invoke_bedrock_with_retry(prompt, max_retries=3, initial_delay=1):
    """Invoke Bedrock with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
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
            return response_body['content'][0]['text']
            
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * initial_delay
                time.sleep(wait_time)
                continue
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
        
        # Process query with retry logic
        if "carbon" in user_input.lower() or "emission" in user_input.lower():
            try:
                calc_results = invoke_calculation_agent(user_input)
                response = invoke_bedrock_with_retry(
                    f"Explain these carbon emission calculation results in detail: {calc_results}"
                )
            except Exception as calc_error:
                response = invoke_bedrock_with_retry(
                    f"I apologize, but I encountered an error calculating the exact emissions. However, I can provide general guidance about carbon emissions and energy efficiency. Please explain what someone should know about carbon emissions for {user_input}"
                )
        else:
            response = invoke_bedrock_with_retry(user_input)
        
        # Store conversation
        store_conversation(conversation_id, user_input, response)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': response,
                'conversationId': conversation_id
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
