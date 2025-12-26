using UnityEngine;
using System.Collections;

namespace two{
public class Gem : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    [Header("Settings")]
    public GameObject objectToSpawn; // Drag the prefab here
    public float maxDistance = 3.4f; 
    public static float posX;
    public static float posY;
    public static float done = 0f;
     private bool canGiveReward = true;

    void Start()
    {
        SpawnObject();
    }

    public void SpawnObject()
    {
        // 1. Calculate a random X that is strictly > 3.4 or < -3.4
        float posX = Random.Range(-maxDistance, maxDistance);

        // 2. Calculate a random Y that is strictly > 3.4 or < -3.4
        float posY = Random.Range(-maxDistance, maxDistance);

        // 3. Create the position vector
        Vector2 spawnPosition = new Vector2(posX, posY);

        // 4. Spawn the object
        // If "objectToSpawn" is null, we assume this script is ON the object we want to move/copy
        if (objectToSpawn != null)
        {
            Instantiate(objectToSpawn, spawnPosition, Quaternion.identity);
        }
    }

    void ontriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Player"))
        {
            Player_Raycast.reward += 3.0f;
            Player_Raycast.done = 1f;
            Destroy(gameObject);
            SpawnObject();
            StartCoroutine(GiveReward());
            StartCoroutine(Done());
        }
    }

     private IEnumerator GiveReward()
    {
        canGiveReward = false;  // prevent multiple rewards immediately
        Debug.Log($"Reward: {Player_Raycast.reward}");

        yield return new WaitForSeconds(0.070f);
        Player_Raycast.reward = -0.04f;

        // Wait for 0.5 seconds
        yield return new WaitForSeconds(0.16f);

        canGiveReward = true;
    }
    private IEnumerator Done()
    {
        yield return new WaitForSeconds(0.070f);
        done = 0;
    }
    }
}