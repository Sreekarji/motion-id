import urllib.request, json

BASE = "http://localhost:8000"

def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}", timeout=15)
    return json.loads(r.read())

def post(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=body,
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=15)
    return json.loads(r.read())

print("=" * 50)
print("Motion ID Pipeline Verification")
print("=" * 50)

# Health
h = get("/health")
print(f"\nGPU:         {h['gpu']} ({h.get('gpu_name', 'N/A')})")
print(f"Users loaded: {h['users_loaded']}")
print(f"MPI models:  {h['mpi_models']}")

# Demo for all 11 users
print("\nUser-by-user results:")
print(f"{'User':<8} {'Decision':<10} {'UV Score':<12} {'Threshold':<12} {'MPI Unlock'}")
print("-" * 55)

users = get("/users")["users"]
for uid in users:
    d = post(f"/predict/demo/{uid}")
    uv = d.get("uv") or {}
    mpi = d.get("mpi") or {}
    print(f"{uid:<8} {d['final_decision']:<10} {str(uv.get('score','N/A')):<12} {str(uv.get('threshold','N/A')):<12} {mpi.get('is_unlock','N/A')}")

print("\n" + "=" * 50)
print("All endpoints responding. Pipeline is working.")
print("=" * 50)
