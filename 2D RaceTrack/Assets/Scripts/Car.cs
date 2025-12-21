using System;
using System.Collections;
using UnityEngine;

public class Car : MonoBehaviour
{
    public static float Time_Rimaining = 10f;
    public static int score = 0;
    public static int done = 0;
    //private bool canGiveReward = true;
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
    public void ManualUpdate(float dt)
    {
        // Decrement time by the fixed step size
        Time_Rimaining -= dt;
        
        if (Time_Rimaining <= 0f)
        {
            ResetCar();
            CarRaycastSensor2D.reward = -4f;
            done = 1;
            Time_Rimaining = 15f; 
        }
    }

    private void ResetCar()
    {
        transform.position = new Vector2(-9.24f, -0.48f); 
        if(rb == null) rb = GetComponent<Rigidbody2D>();
        rb.linearVelocity = Vector2.zero;
        transform.rotation = Quaternion.Euler(0, 0, 0); 
        score = 0;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            ResetCar();
            CarRaycastSensor2D.reward = -1.5f;
            done = 1;
            Time_Rimaining = 15f; 
        }
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Goal"))
        {
            Goals goal = other.GetComponent<Goals>();
            // Simple logic without coroutines
            if (goal.goalNumber == score)
            {
                score++;
                CarRaycastSensor2D.reward = 2.5f + (2 * score);
                Time_Rimaining += 5f;
            }
            else if (goal.goalNumber < score)
            {
                CarRaycastSensor2D.reward = -(1.2f + (score * score));
            }
            else if (goal.goalNumber > score)
            {
                CarRaycastSensor2D.reward = -2.8f;
            }
        }
    }
}
