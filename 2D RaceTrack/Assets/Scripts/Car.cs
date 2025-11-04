using System;
using System.Collections;
using UnityEngine;

public class Car : MonoBehaviour
{
    public static int score = 0;
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

    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            transform.position = new Vector2(-9.24f, -0.48f); // Reset position on collision with wall
            rb.linearVelocity = Vector2.zero;
            transform.rotation = Quaternion.Euler(0, 0, 0); // Reset rotation
            score = 0;
            CarRaycastSensor2D.reward = -1.0f;
            done = 1;
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
                    CarRaycastSensor2D.reward = 2.8f;
                    StartCoroutine(GiveReward());
                }
            }
            else if (goal.goalNumber < score)
            {
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = -0.6f;
                    StartCoroutine(GiveReward());
                }
            }
            else if (goal.goalNumber > score)
            {
                if (canGiveReward)
                {
                    CarRaycastSensor2D.reward = -0.8f;
                    StartCoroutine(GiveReward());
                }
            }
        }
    }

    private IEnumerator GiveReward()
    {
        canGiveReward = false;  // prevent multiple rewards immediately
        Debug.Log($"Reward: {CarRaycastSensor2D.reward}");

        yield return new WaitForSeconds(0.04f);
        CarRaycastSensor2D.reward = -0.01f;

        // Wait for 0.5 seconds
        yield return new WaitForSeconds(0.16f);

        canGiveReward = true;
    }
    private IEnumerator Done()
    {
        yield return new WaitForSeconds(0.04f);
        done = 0;
    }
    
}
