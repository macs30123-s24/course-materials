import json

def is_prime(n):
    """Check if n is a prime number."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def lambda_handler(event, context):
    N = event.get('N')
    primes = [n for n in range(N) if is_prime(n)]
    return {
        'statusCode': 200,
        'body': primes
    }