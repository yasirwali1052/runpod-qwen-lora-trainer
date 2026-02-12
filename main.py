import requests
import json
import os
from datetime import datetime
import time

CONFIG_FILE = "config.json"
API_URL = "https://api.runpod.io/graphql"
CUSTOM_IMAGE = "yasirkhan1052/qwen-api:latest"

def save_config(api_key, gpu):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"api_key": api_key, "default_gpu": gpu}, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def log(msg):
    os.makedirs("logs", exist_ok=True)
    with open("logs/activity.log", "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def graphql(api_key, query):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    resp = requests.post(API_URL, json={"query": query}, headers=headers)
    return resp.json()

def get_available_gpus(api_key):
    query = "{ gpuTypes { id displayName memoryInGb } }"
    result = graphql(api_key, query)
    return result.get('data', {}).get('gpuTypes', [])

def create_pod(api_key, gpu_id, name, disk_gb):
    query = f"""
    mutation {{
        podFindAndDeployOnDemand(
            input: {{
                cloudType: SECURE,
                gpuCount: 1,
                volumeInGb: 0,
                containerDiskInGb: {int(disk_gb)},
                minVcpuCount: 2,
                minMemoryInGb: 15,
                gpuTypeId: "{gpu_id}",
                name: "{name}",
                imageName: "{CUSTOM_IMAGE}",
                dockerArgs: "",
                ports: "8000/http",
                volumeMountPath: "/workspace",
                env: []
            }}
        ) {{ 
            id 
            name 
            runtime {{
                ports {{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                }}
            }}
        }}
    }}
    """
    
    result = graphql(api_key, query)
    
    if "errors" in result:
        log(f"ERROR: {result['errors']}")
        print(f"Error: {result['errors'][0]['message']}")
        return None
    
    pod = result['data']['podFindAndDeployOnDemand']
    log(f"Created pod {pod['id']}")
    return pod

def get_pods(api_key):
    query = """{ 
        myself { 
            pods { 
                id 
                name 
                desiredStatus 
                machine { 
                    gpuDisplayName 
                }
                runtime {
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
            } 
        } 
    }"""
    result = graphql(api_key, query)
    return result.get('data', {}).get('myself', {}).get('pods', [])

def get_pod_endpoint(api_key, pod_id):
    pods = get_pods(api_key)
    pod = next((p for p in pods if p['id'] == pod_id), None)
    
    if not pod or not pod.get('runtime') or not pod['runtime'].get('ports'):
        return None
    
    for port in pod['runtime']['ports']:
        if port['privatePort'] == 8000 and port.get('publicPort'):
            ip = port.get('ip', '')
            public_port = port.get('publicPort')
            return f"http://{ip}:{public_port}"
    
    return None

def stop_pod(api_key, pod_id):
    query = f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id }} }}'
    result = graphql(api_key, query)
    if "errors" in result:
        print(f"Error: {result['errors'][0]['message']}")
    else:
        log(f"Stopped pod {pod_id}")
        print(f"Pod stopped: {pod_id}")

def terminate_pod(api_key, pod_id):
    query = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    result = graphql(api_key, query)
    if "errors" in result:
        print(f"Error: {result['errors'][0]['message']}")
    else:
        log(f"Terminated pod {pod_id}")
        print(f"Pod terminated: {pod_id}")

def process_images_on_pod(api_key, pod_id):
    endpoint = get_pod_endpoint(api_key, pod_id)
    
    if not endpoint:
        print("Pod API endpoint not ready")
        return
    
    input_folder = "data/input"
    output_folder = "data/output"
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        print("No images in data/input/")
        return
    
    print(f"Processing {len(image_files)} images via {endpoint}")
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, img_file)
        
        print(f"[{i}/{len(image_files)}] {img_file}")
        
        try:
            with open(img_path, 'rb') as f:
                response = requests.post(
                    f"{endpoint}/process",
                    files={'image': (img_file, f, 'image/png')},
                    timeout=120
                )
            
            if response.status_code == 200:
                result = response.json()
                
                output_file = os.path.join(
                    output_folder,
                    f"{os.path.splitext(img_file)[0]}.json"
                )
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"   Saved: {output_file}")
            else:
                print(f"   Error: {response.status_code}")
        
        except Exception as e:
            print(f"   Failed: {str(e)}")
    
    print(f"\nResults saved in {output_folder}/")

def main():
    config = load_config()
    
    if not config:
        api_key = input("Enter RunPod API Key: ")
        save_config(api_key, "NVIDIA RTX A4000")
        config = load_config()
    
    api_key = config['api_key']
    
    while True:
        print("\n" + "="*50)
        print("QWEN RUNPOD MANAGER")
        print("="*50)
        print("1. Create Pod")
        print("2. View Pods & Endpoints")
        print("3. Process Images")
        print("4. Stop Pod")
        print("5. Terminate Pod")
        print("6. Exit")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "1":
            print("\nFetching GPUs...")
            gpus = get_available_gpus(api_key)
            
            print("\nAvailable GPUs:")
            for i, g in enumerate(gpus[:20], 1):
                print(f"{i}. {g['displayName']} ({g['memoryInGb']}GB)")
            
            try:
                gpu_index = int(input("\nSelect GPU number: ")) - 1
                selected_gpu_id = gpus[gpu_index]['id']
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")
                continue

            name = input("Pod name [Qwen-Worker]: ") or "Qwen-Worker"
            disk = input("Disk GB [50]: ") or "50"
            
            pod = create_pod(api_key, selected_gpu_id, name, int(disk))
            if pod:
                print(f"\nPod created: {pod['id']}")
                print("Wait 5-10 minutes for model to download and load into VRAM")
        
        elif choice == "2":
            pods = get_pods(api_key)
            print(f"\n{'ID':<18} {'NAME':<20} {'STATUS':<12} {'GPU'}")
            print("-" * 80)
            
            for p in pods:
                gpu = p['machine']['gpuDisplayName'] if p.get('machine') else "INIT"
                print(f"{p['id']:<18} {p['name']:<20} {p['desiredStatus']:<12} {gpu}")
                
                endpoint = get_pod_endpoint(api_key, p['id'])
                if endpoint:
                    print(f"  API: {endpoint}")
                else:
                    print(f"  API: Not ready (Wait for model download)")
                print()
        
        elif choice == "3":
            pod_id = input("Enter Pod ID: ")
            process_images_on_pod(api_key, pod_id)
        
        elif choice == "4":
            pod_id = input("Enter Pod ID: ")
            stop_pod(api_key, pod_id)
        
        elif choice == "5":
            pod_id = input("Enter Pod ID: ")
            terminate_pod(api_key, pod_id)
        
        elif choice == "6":
            break

if __name__ == "__main__":
    main()