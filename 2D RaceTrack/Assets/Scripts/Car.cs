using System;
using UnityEngine;

public class Car : MonoBehaviour
{
    public static int score = 0;
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
            Debug.Log("Score: " + score);
        }
        else if (goal.goalNumber < score)
        {
            Debug.Log("You already passed this goal. Current Score: " + score);
        }
    }
}
}
