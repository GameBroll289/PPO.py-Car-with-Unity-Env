using System;
using System.Collections;
using UnityEngine;

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
            CarRaycastSensor2D.reward = -1f;
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
            CarRaycastSensor2D.reward = -5f;
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
            Goals goal = other.GetComponent<Goals>();
            if (goal.goalNumber == score)
            {
                score++;
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = 10+(score*0.5f);
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

        yield return new WaitForSeconds(0.079f);
        CarRaycastSensor2D.reward = -0.04f;

        // Wait for 0.5 seconds
        yield return new WaitForSeconds(0.16f);

        canGiveReward = true;
    }
    private IEnumerator Done()
    {
        yield return new WaitForSeconds(0.079f);
        done = 0;
    }
    
}
