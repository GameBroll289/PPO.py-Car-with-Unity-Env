using UnityEngine;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using JetBrains.Annotations;

namespace two{
public class Player_Raycast : MonoBehaviour
{   
    public static float reward = -0.02f;
    public float rayLength = 10f;
    public LayerMask obstacleMask;
    public static float done = 0f;
    public static float[] obs = new float[4]; // Goal x & y, Player

    [HideInInspector]
    public float[] WallDistances = new float[4];

    private Vector2[] localDirections = new Vector2[4]
    {
    new Vector2( 0f,  1f),                 // 0°   Front (Up)
    //new Vector2( 0.70710678f,  0.70710678f), // 45°   Front-Right
    new Vector2( 1f,  0f),                 // 90°   Right
    new Vector2( 0f, -1f),                 // 180°  Back (Down)
    new Vector2(-1f,  0f),                 // 270°  Left
    };

    // Memory Mapped File variables
    const string memoryName = "unity_ram2";
    const int slotCount = 12;   // must match Python
    private const int slotSize = 4;     // float32
    private const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;
    void Awake() {
    obstacleMask = LayerMask.GetMask("Wall");
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
            RaycastHit2D hitWall = Physics2D.Raycast(transform.position, direction, rayLength, obstacleMask);

            WallDistances[i] = hitWall.distance / rayLength;
            //Blue ray if something is hit
            Debug.DrawRay(transform.position, direction * hitWall.distance, Color.white);
        
            if (hitWall.collider == null)
            {
                WallDistances[i] = -1f;
                //Red ray if nothing is hit
                Debug.DrawRay(transform.position, direction * rayLength, Color.black);
            }
        }
        obs[0] = transform.position.x;
        obs[1] = transform.position.y;
        obs[2] = Gem.posX;
        obs[3] = Gem.posY;
        // Write state to shared memory
        WriteFloats(2, obs);//Goal and player x & y positions
        WriteFloats(6, WallDistances);
        WriteFloat(10, reward); // Cumulative reward
        WriteFloat(11, done);
        accessor.Flush();

        // Debug.Log to console
        // Debug.Log($"R: {CarRaycastSensor2D.reward+speed}");
        // Debug.Log($"{acceleration};{steering};Reward: {reward};{Car.done};{(GetComponent<Rigidbody2D>().linearVelocity.magnitude / 5f)}; WallRays: {string.Join(",", WallDistances)}; Rays: {string.Join(",", rayDistances)}; Hits: {string.Join(";", HitsInfo)}");
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
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               