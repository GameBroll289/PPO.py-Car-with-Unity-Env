using UnityEngine;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using JetBrains.Annotations;


public class CarRaycastSensor2D : MonoBehaviour
{
    public static float speed;
    public static float reward = 0.01f;
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
    const int slotCount = 29;   // must match Python
    const int slotSize = 4;     // float32
    const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;
    void Awake() {
    obstacleMask = LayerMask.GetMask("Raycast", "Wall");
    WallMask = LayerMask.GetMask("Wall");
}

    void Start()
    {
        
        mmf = MemoryMappedFile.CreateOrOpen(memoryName, totalSize, MemoryMappedFileAccess.ReadWrite);


        accessor = mmf.CreateViewAccessor(0, totalSize, MemoryMappedFileAccess.ReadWrite);
    }

    void FixedUpdate()
    {
        float[] HitsInfo = new float[localDirections.Length]; // All elements are 0 by default

        for (int i = 0; i < localDirections.Length; i++)
        {
            Vector2 direction = transform.TransformDirection(localDirections[i]);
            RaycastHit2D hit = Physics2D.Raycast(transform.position, direction, rayLength, obstacleMask);
            RaycastHit2D hitWall = Physics2D.Raycast(transform.position, direction, rayLength, WallMask);

            if (hit.collider != null)
            {
                //If hits current goal "-1" and if previous than "-0.5" and if one of the next goal than "-0.3"
                if (hit.collider.CompareTag("Goal"))
                {
                    Goals goal = hit.collider.GetComponent<Goals>();

                    if (goal.goalNumber == Car.score)
                    {
                        HitsInfo[i] = -1;
                    }
                    else if (goal.goalNumber < Car.score)
                    {
                        HitsInfo[i] = -0.5f;
                    }
                    else if (goal.goalNumber > Car.score)
                    {
                        HitsInfo[i] = -0.3f;
                    }
                }

                rayDistances[i] = hit.distance / rayLength;
                WallDistances[i] = hitWall.distance / rayLength;
                //Blue ray if something is hit
                Debug.DrawRay(transform.position, direction * hitWall.distance, Color.chocolate);
                Debug.DrawRay(transform.position, direction * hit.distance, Color.blue);
            }
            else
            {
                rayDistances[i] = 1f;
                WallDistances[i] = 1f;
                //Red ray if nothing is hit
                Debug.DrawRay(transform.position, direction * rayLength, Color.darkRed);
                Debug.DrawRay(transform.position, direction * rayLength, Color.red);
            }
        }

        speed = GetComponent<Rigidbody2D>().linearVelocity.magnitude / 11f;

        // Write state to shared memory
        WriteFloats(0, rayDistances);
        WriteFloats(8, HitsInfo);
        WriteFloat(16, reward); // Cumulative reward
        WriteFloat(17, Car.done);
        WriteFloat(20, (speed)); // Speed normalized
        WriteFloats(21, WallDistances);
        accessor.Flush();

        // Read actions back from Python
        float acceleration = ReadFloat(18); //3
        float steering = ReadFloat(19); //4

        // Debug.Log to console
        Debug.Log($"R: {reward}");
        //Debug.Log($"{acceleration};{steering};Reward: {reward};{Car.done};{(GetComponent<Rigidbody2D>().linearVelocity.magnitude / 5f)}; WallRays: {string.Join(",", WallDistances)}; Rays: {string.Join(",", rayDistances)}; Hits: {string.Join(";", HitsInfo)}");
        //Car.done = 0;
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
