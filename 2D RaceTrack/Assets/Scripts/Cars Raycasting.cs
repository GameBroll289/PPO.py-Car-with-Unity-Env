using UnityEngine;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;


public class CarRaycastSensor2D : MonoBehaviour
{
 
    public static float reward = 0f;
    public float rayLength = 10f;
    public LayerMask obstacleMask;

    [HideInInspector]
    public float[] rayDistances = new float[8];

    private Vector2[] localDirections = new Vector2[8]
    {
        Vector2.up,                                      // Front
        (Vector2.up + Vector2.right).normalized,         // Front-Right
        Vector2.right,                                   // Right
        (-Vector2.up + Vector2.right).normalized,        // Back-Right
        Vector2.down,                                    // Back
        (-Vector2.up - Vector2.right).normalized,        // Back-Left
        Vector2.left,                                    // Left
        (Vector2.up - Vector2.right).normalized          // Front-Left
    };
    private string filePath;


    // Memory Mapped File variables
    const string memoryName = "unity_ram";
    const int slotCount = 21;   // must match Python
    const int slotSize = 4;     // float32
    const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;

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
                //if (Input.GetKey(KeyCode.Space)) // Only draw when space is held
                Debug.DrawRay(transform.position, direction * hit.distance, Color.blue);
            }
            else
            {
                rayDistances[i] = 1f;
                //if (Input.GetKey(KeyCode.Space)) // Only draw when space is held
                Debug.DrawRay(transform.position, direction * rayLength, Color.red);
            }
        }

        

        // Write state to shared memory
        WriteFloats(0, rayDistances);
        WriteFloats(8, HitsInfo);
        WriteFloat(16, reward);
        WriteFloat(17, Car.done);
        WriteFloat(20, (GetComponent<Rigidbody2D>().linearVelocity.magnitude / 5f)); // Speed normalized
        accessor.Flush();

        // Read actions back from Python
        float acceleration = ReadFloat(18); //3
        float steering = ReadFloat(19); //4

        // Debug.Log to console
        Debug.Log($"{acceleration};{steering};Reward: {reward};{Car.done};{(GetComponent<Rigidbody2D>().linearVelocity.magnitude / 5f)}; Rays: {string.Join(",", rayDistances)}; Hits: {string.Join(";", HitsInfo)}");
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
