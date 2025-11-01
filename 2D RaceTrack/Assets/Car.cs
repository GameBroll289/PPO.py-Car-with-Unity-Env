using UnityEngine;

public class RaceCar : MonoBehaviour
{
    public Rigidbody2D rb;
    public float speed=5;
    public float turnSpeed=100;
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
        float move = Input.GetAxis("Vertical") * speed * Time.deltaTime;
        float turn = Input.GetAxis("Horizontal") * turnSpeed * Time.deltaTime;

        rb.AddForce(transform.up * move, ForceMode2D.Impulse);
        rb.MoveRotation(rb.rotation - turn);
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
}
