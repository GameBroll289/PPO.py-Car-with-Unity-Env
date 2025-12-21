using UnityEngine;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using JetBrains.Annotations;

public class CarRaycastSensor2D : MonoBehaviour
{
    public float StartingTime = 15f;
    public static float speed;
    public static float reward = -0.02f;
    public float rayLength = 10f;
    public LayerMask obstacleMask, WallMask;

    [HideInInspector]
    public float[] rayDistances = new float[8];
    public float[] WallDistances = new float[8];

    private Vector2[] localDirections = new Vector2[8]
    {
    new Vector2( 0f,  1f),                 // 0°   Front (Up)
    new Vector2( 0.38268343f,  0.92387953f), // 22.5°  Front–FrontRight
    //new Vector2( 0.70710678f,  0.70710678f), // 45°   Front-Right
    //new Vector2( 0.92387953f,  0.38268343f), // 67.5° FrontRight–Right
    new Vector2( 1f,  0f),                 // 90°   Right
    //new Vector2( 0.92387953f, -0.38268343f), // 112.5° Right–BackRight
    //new Vector2( 0.70710678f, -0.70710678f), // 135°  Back-Right
    new Vector2( 0.38268343f, -0.92387953f), // 157.5° BackRight–Back
    new Vector2( 0f, -1f),                 // 180°  Back (Down)
    new Vector2(-0.38268343f, -0.92387953f), // 202.5° Back–BackLeft
    //new Vector2(-0.70710678f, -0.70710678f), // 225°  Back-Left
    //new Vector2(-0.92387953f, -0.38268343f), // 247.5° BackLeft–Left
    new Vector2(-1f,  0f),                 // 270°  Left
    //new Vector2(-0.92387953f,  0.38268343f), // 292.5° Left–FrontLeft
    //new Vector2(-0.70710678f,  0.70710678f), // 315°  Front-Left
    new Vector2(-0.38268343f,  0.92387953f),  // 337.5° FrontLeft–Front
    };
    // Memory Mapped File variables
    const string memoryName = "unity_ram";
    const int slotCount = 31;   // must match Python
    const int slotSize = 4;     // float32
    const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;
    // --- Script References ---
    Car carScript;
    AICarController aiController;

    void Awake() 
    {
        obstacleMask = LayerMask.GetMask("Raycast", "Wall");
        WallMask = LayerMask.GetMask("Wall");
        
        carScript = GetComponent<Car>();
        aiController = GetComponent<AICarController>();

        // IMPORTANT: We take control of the Physics Clock
        Physics2D.simulationMode = SimulationMode2D.Script;
    }

    void Start()
    {
        mmf = MemoryMappedFile.CreateOrOpen(memoryName, totalSize, MemoryMappedFileAccess.ReadWrite);
        accessor = mmf.CreateViewAccessor(0, totalSize, MemoryMappedFileAccess.ReadWrite);
        
        // Clear the sync flag on start
        WriteFloat(30, 0f);
    }

    void Update()
    {
        // 1. Check if Python has sent a command (Sync == 1)
        float syncState = ReadFloat(30);
        
        if (syncState == 1f)
        {
            // === IT IS UNITY'S TURN ===
            float fixedDt = 0.02f;

            // A. Reset per-step variables
            Car.done = 0; 
            reward = -0.02f; 
            
            // B. Read Actions from Python
            float moveAction = ReadFloat(18);
            float turnAction = ReadFloat(19);
            
            // C. Apply Forces via your AI Controller
            if (aiController != null)
            {
                aiController.ManualMove(moveAction, turnAction, fixedDt);
            }

            // D. Step Physics Manually
            Physics2D.Simulate(fixedDt);
            
            // E. Update Game Logic (Timer, Score checks)
            carScript.ManualUpdate(fixedDt);

            // F. Gather Observations (Raycasts)
            PerformRaycasts();
            speed = GetComponent<Rigidbody2D>().linearVelocity.magnitude / 11f;

            // G. Write State back to MMF
            WriteFloats(0, rayDistances);
            WriteFloats(8, GetHitsInfo());
            WriteFloat(16, reward + speed); 
            WriteFloat(17, Car.done);
            WriteFloat(20, speed);
            WriteFloat(21, Car.Time_Rimaining / carScript.speed);
            WriteFloats(22, WallDistances);
            
            // H. HANDSHAKE: Tell Python we are done
            WriteFloat(30, 0f); 
            
            accessor.Flush();
        }
    }
    
    // --- Helper Methods ---
    void PerformRaycasts() {
        for (int i = 0; i < localDirections.Length; i++) {
            Vector2 direction = transform.TransformDirection(localDirections[i]);
            RaycastHit2D hit = Physics2D.Raycast(transform.position, direction, rayLength, obstacleMask);
            RaycastHit2D hitWall = Physics2D.Raycast(transform.position, direction, rayLength, WallMask);

            if (hit.collider != null) {
                rayDistances[i] = hit.distance / rayLength;
                WallDistances[i] = hitWall.distance / rayLength;
                Debug.DrawRay(transform.position, direction * hit.distance, Color.blue);
            } else {
                rayDistances[i] = 1f;
                WallDistances[i] = 1f;
                Debug.DrawRay(transform.position, direction * rayLength, Color.red);
            }
        }
    }

    float[] GetHitsInfo() {
        float[] HitsInfo = new float[8];
        for (int i = 0; i < localDirections.Length; i++) {
             Vector2 direction = transform.TransformDirection(localDirections[i]);
             RaycastHit2D hit = Physics2D.Raycast(transform.position, direction, rayLength, obstacleMask);
             if (hit.collider != null && hit.collider.CompareTag("Goal")) {
                Goals goal = hit.collider.GetComponent<Goals>();
                if (goal.goalNumber == Car.score) HitsInfo[i] = -1;
                else if (goal.goalNumber < Car.score) HitsInfo[i] = -0.5f;
                else HitsInfo[i] = -0.3f;
             }
        }
        return HitsInfo;
    }


    void OnApplicationQuit()
    {
        accessor?.Dispose();
        mmf?.Dispose();
    }

    // -----------------------------
    // Helpers
    // -----------------------------
    void WriteFloat(int slot, float value)
    {
        accessor.Write(slot * slotSize, value);
    }

    void WriteFloats(int startSlot, float[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            accessor.Write((startSlot + i) * slotSize, values[i]);
        }
    }

    float ReadFloat(int slot)
    {
        return accessor.ReadSingle(slot * slotSize);
    }
}
