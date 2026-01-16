using UnityEngine;
using System.IO.MemoryMappedFiles;
using System.Collections;
namespace two{
public class Player : MonoBehaviour
{
    public GameObject objectToSpawn; // Drag the prefab here
    public float maxDistance = 3.4f; 
    public static float posX;
    public static float posY;
    public static float done = 0f;
    public static bool canGiveReward = true;

    public float moveSpeed = 2f;
    public static float Time_Rimaining=20f;
    float StartingTime=20f;

    private const float SIGNAL_CODE = -999.0f;
    const string memoryName = "unity_ram2";
    const int slotCount = 30;   // must match Python
    const int slotSize = 4;     // float32
    const int totalSize = slotCount * slotSize;

    MemoryMappedFile mmf;
    MemoryMappedViewAccessor accessor;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        mmf = MemoryMappedFile.CreateOrOpen(memoryName, totalSize, MemoryMappedFileAccess.ReadWrite);
        accessor = mmf.CreateViewAccessor(0, totalSize, MemoryMappedFileAccess.ReadWrite);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Time_Rimaining -= Time.deltaTime;
        Debug.Log($"Time Remaining: {Time_Rimaining}");
        if (Time_Rimaining <= 0f)
        {
            Player_Raycast.reward = -5f;
            Player_Raycast.done = 1;
            StartCoroutine(GiveReward());
            StartCoroutine(Done());
        }

        float vertical = ReadSlot(0);
        float horizontal = ReadSlot(1);

        Debug.Log($"R: {Player_Raycast.reward}; D: {Player_Raycast.done}");
        // IF PYTHON HASN'T SENT A NEW COMMAND, WAIT.
        if (vertical == SIGNAL_CODE) return;
        Vector2 movement = new Vector2(horizontal, vertical);
        transform.Translate(movement * moveSpeed * Time.fixedDeltaTime);

        // TELL PYTHON: "I FINISHED THIS FRAME"
        WriteFloat(0, SIGNAL_CODE);
    }

    float ReadSlot(int index)
    {
        float value;
        accessor.Read(index * slotSize, out value);
        return value;
    }

    void OnApplicationQuit()
    {
        accessor?.Dispose();
        mmf?.Dispose();
    }

    void WriteFloat(int slot, float value)
    {
        accessor.Write(slot * slotSize, value);
    }

    public void SpawnObject()
    {
        // 1. Calculate a random X that is strictly > 3.4 or < -3.4
        posX = Random.Range(-maxDistance, maxDistance);

        // 2. Calculate a random Y that is strictly > 3.4 or < -3.4
        posY = Random.Range(-maxDistance, maxDistance);

        // 3. Create the position vector
        Vector2 spawnPosition = new Vector2(posX, posY);

        // 4. Spawn the object
        // If "objectToSpawn" is null, we assume this script is ON the object we want to move/copy
        Instantiate(objectToSpawn, spawnPosition, Quaternion.identity);
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Wall"))
        {
            Player_Raycast.reward = -15f;
            Player_Raycast.done = 1f;
            
            StartCoroutine(GiveReward());
            StartCoroutine(Done());
        }

        if (other.CompareTag("Gem"))
        {
            Player_Raycast.reward = 50f;
            Player_Raycast.done = 0f;
            Destroy(other.gameObject);
            SpawnObject();
            StartCoroutine(GiveReward());
            Time_Rimaining+=10f; // Reset time remaining
        }
    }

     private IEnumerator GiveReward()
    {
        canGiveReward = false;  // prevent multiple rewards immediately
        Debug.Log($"Reward: {Player_Raycast.reward}");

        yield return new WaitForSeconds(0.070f);
        Player_Raycast.reward = 0f;

        // Wait for 0.5 seconds
        yield return new WaitForSeconds(0.16f);

        canGiveReward = true;
    }
    private IEnumerator Done()
    {
        yield return new WaitForSeconds(0.070f);
        Player_Raycast.done = 0;
        Time_Rimaining = StartingTime; // Reset time remaining
        transform.position = new Vector2(0f, 0f); // Reset position on collision with wall
        //DEstry all gems
        GameObject[] gems = GameObject.FindGameObjectsWithTag("Gem");
        foreach (GameObject gem in gems)
        {
            Destroy(gem);
        }
        SpawnObject(); // Respawn gem
        Debug.Log($"Done reset to {Player_Raycast.done}");
    }}
}