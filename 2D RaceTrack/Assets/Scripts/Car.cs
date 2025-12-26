using System;
using System.Collections;
using UnityEngine;

namespace one{
public class Car : MonoBehaviour
{
    public static float Time_Rimaining = 60f;
    public static int score = 2;
    public static int done = 0;
    private bool canGiveReward = true;
    public Rigidbody2D rb;
    public float speed = 5;
    public float turnSpeed = 100;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (rb == null)
        {
            rb = GetComponent<Rigidbody2D>();
        }
    }

    // Update is called once per frame
    void Update()
    {
        Time_Rimaining -= Time.deltaTime;
        Debug.Log($"Time Remaining: {Time_Rimaining}");
        if (Time_Rimaining <= 0f)
        {
            transform.position = new Vector2(-6.8f, 2.98f); // Reset position on time out
            rb.linearVelocity = Vector2.zero;
            transform.rotation = Quaternion.Euler(0, 0, -88.906f); // Reset rotation
            score = 2;
            CarRaycastSensor2D.reward = -20f;
            done = 1;
            StartCoroutine(GiveReward());
            StartCoroutine(Done());
            Time_Rimaining = 60f; // Reset time remaining
        }
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            transform.position = new Vector2(-6.8f, 2.98f); // Reset position on collision with wall
            rb.linearVelocity = Vector2.zero;
            transform.rotation = Quaternion.Euler(0, 0, -88.906f); // Reset rotation
            score = 2;
            CarRaycastSensor2D.reward = -15f;
            done = 1;
            Time_Rimaining = 60f; // Reset time remaining
            StartCoroutine(GiveReward());
            StartCoroutine(Done());
        }
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Goal"))
        {
            // [NEW] Make the goal invisible to raycasts instantly
            other.gameObject.layer = 2; // Layer 2 is 'Ignore Raycast' by default in Unity

            Goals goal = other.GetComponent<Goals>();
            if (goal.goalNumber == score)
            {
                score++;
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = score*3f;
                    Time_Rimaining += 15f;
                    StartCoroutine(GiveReward());
                }
            }
            else if (goal.goalNumber < score)
            {
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = -(1f);
                    StartCoroutine(GiveReward());
                }
            }
            else if (goal.goalNumber > score)
            {
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = -2f;
                    StartCoroutine(GiveReward());
                }
            }
        }
 } 

    private IEnumerator GiveReward()
    {
        canGiveReward = false;  // prevent multiple rewards immediately
        Debug.Log($"Reward: {CarRaycastSensor2D.reward}");

        yield return new WaitForSeconds(0.070f);
        CarRaycastSensor2D.reward = -0.04f;

        // Wait for 0.5 seconds
        yield return new WaitForSeconds(0.16f);

        canGiveReward = true;
    }
    private IEnumerator Done()
    {
        yield return new WaitForSeconds(0.070f);
        done = 0;
    }
    
    // [NEW] Add this function to restore the layer when you leave the goal
    void OnTriggerExit2D(Collider2D other)
    {
        if (other.CompareTag("Goal"))
        {
            // Restore the goal to the 'Raycast' layer (assuming Raycast layer is index 6 or 7?)
            // Use the Layer ID your goals are normally on (e.g., 0 for Default, 6 for Raycast)
            other.gameObject.layer = LayerMask.NameToLayer("Raycast"); 
            // If "Raycast" is not the exact name, use the integer ID directly, e.g., other.gameObject.layer = 6;
        }
    }
}}
