import os
from src.agent import stream_agent, AgentState

# We will run this without a valid Google API key to test the graceful degradation
# and ensure the pipeline runs successfully end-to-end.

def main():
    print("Starting TaskPilot AI Pipeline Test...")
    
    events = stream_agent("Generate my daily plan", state=None)
    
    for event in events:
        if event["type"] == "tool_call":
            print(f"🔧 Tool called: {event['content']['name']}")
        elif event["type"] == "tool_result":
            print(f"✅ Tool result: {event['content']['name']}")
        elif event["type"] == "response":
            print(f"\n🤖 Agent Response:\n{event['content']}")
            
if __name__ == "__main__":
    main()
