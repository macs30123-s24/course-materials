import boto3
import json

def make_def(lambda_arn):
    definition = {
      "Comment": "Q2 State Machine",
      "StartAt": "Map",
      "States": {
        "Map": {
          "Type": "Map",
          "End": True,
          "MaxConcurrency": 10,
          "Iterator": {
            "StartAt": "Lambda Invoke",
            "States": {
              "Lambda Invoke": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "OutputPath": "$.Payload",
                "Parameters": {
                  "Payload.$": "$",
                  "FunctionName": lambda_arn
                },
                "Retry": [
                  {
                    "ErrorEquals": [
                      "Lambda.ServiceException",
                      "Lambda.AWSLambdaException",
                      "Lambda.SdkClientException",
                      "Lambda.TooManyRequestsException",
                      "States.TaskFailed",
                      "Lambda.Unknown"                      
                    ],
                    "IntervalSeconds": 2,
                    "MaxAttempts": 6,
                    "BackoffRate": 2
                  }
                ],
                "End": True
              }
            }
          }
        }
      }
    }
    return definition

if __name__ == '__main__':
    iam = boto3.client('iam')
    sfn = boto3.client('stepfunctions')
    aws_lambda = boto3.client('lambda')
    role = iam.get_role(RoleName='LabRole')

    lambda_function_name = input("What is the name of the Lambda function you would like to invoke in your Step Functions Map State? (default: 'q2'): ")
    if len(lambda_function_name) == 0:
        lambda_function_name = 'q2'

    # Get Lambda Function ARN and Role ARN
    # Assumes Lambda function already exists
    lambda_arn = [f['FunctionArn']
                  for f in aws_lambda.list_functions()['Functions']
                  if f['FunctionName'] == lambda_function_name][0]
    
    # Throttle concurrent executions to 10
    response = aws_lambda.put_function_concurrency(
            FunctionName=lambda_function_name,
            ReservedConcurrentExecutions=10
        )

    sfn_function_name = input("What would like to name your Step Function State Machine? (default: 'q2'): ")
    if len(sfn_function_name) == 0:
        sfn_function_name = 'q2'

    # Use Lambda ARN to create State Machine Definition
    sf_def = make_def(lambda_arn)

    # Create Step Function State Machine if doesn't already exist
    try:
        response = sfn.create_state_machine(
            name=sfn_function_name,
            definition=json.dumps(sf_def),
            roleArn=role['Role']['Arn'],
            type='EXPRESS'
        )
    except sfn.exceptions.StateMachineAlreadyExists:
        response = sfn.list_state_machines()
        state_machine_arn = [sm['stateMachineArn'] 
                            for sm in response['stateMachines'] 
                            if sm['name'] == sfn_function_name][0]
        response = sfn.update_state_machine(
            stateMachineArn=state_machine_arn,
            definition=json.dumps(sf_def),
            roleArn=role['Role']['Arn']
        )
